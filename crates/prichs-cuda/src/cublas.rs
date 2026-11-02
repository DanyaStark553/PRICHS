use anyhow::{bail, Result};
use once_cell::sync::Lazy;
use std::ffi::c_void;
use std::sync::Mutex;

pub const CUBLAS_OP_N: i32 = 0;
pub const CUBLAS_OP_T: i32 = 1;
pub const CUBLAS_OP_C: i32 = 2;

pub const CUBLAS_TF32_TENSOR_OP_MATH: i32 = 3;

pub const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

#[link(name = "cudart")]
extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

#[link(name = "cublas")]
extern "C" {
    fn cublasCreate_v2(handle: *mut *mut c_void) -> i32;
    fn cublasDestroy_v2(handle: *mut c_void) -> i32;
    fn cublasSetMathMode(handle: *mut c_void, mode: i32) -> i32;
    fn cublasSgemm_v2(
        handle: *mut c_void,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: *const f32,
        c: *mut f32,
        ldc: i32,
    ) -> i32;
}

/// Persistent GPU Workspace holding 256 MB VRAM and cuBLAS Handle with TF32 Tensor Cores
pub struct GpuWorkspace {
    handle: *mut c_void,
    dev_base: *mut c_void,
    _total_bytes: usize,
    slab_a_offset: usize,
    slab_a_size: usize,
    slab_b_offset: usize,
    slab_b_size: usize,
    slab_c_offset: usize,
    slab_c_size: usize,
}

unsafe impl Send for GpuWorkspace {}
unsafe impl Sync for GpuWorkspace {}

impl GpuWorkspace {
    /// Initialize 256 MB VRAM Workspace & cuBLAS with TF32 Tensor Cores
    pub fn new() -> Result<Self> {
        let mut handle: *mut c_void = std::ptr::null_mut();
        let status = unsafe { cublasCreate_v2(&mut handle) };
        if status != 0 || handle.is_null() {
            bail!("cublasCreate_v2 failed with status {}", status);
        }

        // Enable TF32 Tensor Cores on Ampere (RTX 3060)
        let math_status = unsafe { cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) };
        if math_status != 0 {
            eprintln!("Warning: cublasSetMathMode(TF32) returned status {}", math_status);
        }

        // Lightweight Fallback VRAM Workspace (64 MB total):
        // Slab A (Input): 16 MB (4M floats)
        // Slab B (Weights / dY): 32 MB (8M floats)
        // Slab C (Output / Grad): 16 MB (4M floats)
        let slab_a_size = 16 * 1024 * 1024;
        let slab_b_size = 32 * 1024 * 1024;
        let slab_c_size = 16 * 1024 * 1024;
        let total_bytes = slab_a_size + slab_b_size + slab_c_size; // 64 MB

        let slab_a_offset = 0;
        let slab_b_offset = slab_a_size;
        let slab_c_offset = slab_a_size + slab_b_size;

        let mut dev_base: *mut c_void = std::ptr::null_mut();
        let cuda_status = unsafe { cudaMalloc(&mut dev_base, total_bytes) };
        if cuda_status != 0 || dev_base.is_null() {
            unsafe { cublasDestroy_v2(handle) };
            bail!("cudaMalloc 64MB fallback failed with status {}", cuda_status);
        }

        Ok(Self {
            handle,
            dev_base,
            _total_bytes: total_bytes,
            slab_a_offset,
            slab_a_size,
            slab_b_offset,
            slab_b_size,
            slab_c_offset,
            slab_c_size,
        })
    }

    #[inline]
    fn dev_ptr(&self, offset: usize) -> *mut f32 {
        unsafe { (self.dev_base as *mut u8).add(offset) as *mut f32 }
    }

    /// Forward Matmul on GPU: C = A * B
    /// A is [m, k], B is [k, n], C is [m, n]
    pub fn matmul(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) -> Result<()> {
        let a_bytes = m * k * std::mem::size_of::<f32>();
        let b_bytes = k * n * std::mem::size_of::<f32>();
        let c_bytes = m * n * std::mem::size_of::<f32>();

        if a_bytes > self.slab_a_size || b_bytes > self.slab_b_size || c_bytes > self.slab_c_size {
            bail!("Matrix dimensions exceed preallocated GPU workspace slabs");
        }

        let ptr_a = self.dev_ptr(self.slab_a_offset);
        let ptr_b = self.dev_ptr(self.slab_b_offset);
        let ptr_c = self.dev_ptr(self.slab_c_offset);

        unsafe {
            // Copy A and B to GPU
            cudaMemcpy(ptr_a as *mut c_void, a.as_ptr() as *const c_void, a_bytes, CUDA_MEMCPY_HOST_TO_DEVICE);
            cudaMemcpy(ptr_b as *mut c_void, b.as_ptr() as *const c_void, b_bytes, CUDA_MEMCPY_HOST_TO_DEVICE);

            let alpha = 1.0f32;
            let beta = 0.0f32;

            // cuBLAS col-major GEMM: C^T = B^T * A^T
            let status = cublasSgemm_v2(
                self.handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                ptr_b,
                n as i32,
                ptr_a,
                k as i32,
                &beta,
                ptr_c,
                n as i32,
            );

            if status != 0 {
                bail!("cublasSgemm_v2 failed with status {}", status);
            }

            // Copy result C back to Host
            cudaMemcpy(c.as_mut_ptr() as *mut c_void, ptr_c as *const c_void, c_bytes, CUDA_MEMCPY_DEVICE_TO_HOST);
            cudaDeviceSynchronize();
        }

        Ok(())
    }

    /// Weight gradient Matmul on GPU: dW += A^T * B
    /// A is [m, k], B is [m, n], dW is [k, n]
    pub fn matmul_at_b(&self, a: &[f32], b: &[f32], dw: &mut [f32], m: usize, k: usize, n: usize) -> Result<()> {
        let a_bytes = m * k * std::mem::size_of::<f32>();
        let b_bytes = m * n * std::mem::size_of::<f32>();
        let dw_bytes = k * n * std::mem::size_of::<f32>();

        if a_bytes > self.slab_a_size || b_bytes > self.slab_b_size || dw_bytes > self.slab_c_size {
            bail!("Matrix dimensions exceed preallocated GPU workspace slabs");
        }

        let ptr_a = self.dev_ptr(self.slab_a_offset);
        let ptr_b = self.dev_ptr(self.slab_b_offset);
        let ptr_c = self.dev_ptr(self.slab_c_offset);

        unsafe {
            cudaMemcpy(ptr_a as *mut c_void, a.as_ptr() as *const c_void, a_bytes, CUDA_MEMCPY_HOST_TO_DEVICE);
            cudaMemcpy(ptr_b as *mut c_void, b.as_ptr() as *const c_void, b_bytes, CUDA_MEMCPY_HOST_TO_DEVICE);

            let alpha = 1.0f32;
            let beta = 0.0f32;

            // cuBLAS col-major GEMM: dW^T = B^T * A
            let status = cublasSgemm_v2(
                self.handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                n as i32,
                k as i32,
                m as i32,
                &alpha,
                ptr_b,
                n as i32,
                ptr_a,
                k as i32,
                &beta,
                ptr_c,
                n as i32,
            );

            if status != 0 {
                bail!("cublasSgemm_v2 (at_b) failed with status {}", status);
            }

            cudaMemcpy(dw.as_mut_ptr() as *mut c_void, ptr_c as *const c_void, dw_bytes, CUDA_MEMCPY_DEVICE_TO_HOST);
            cudaDeviceSynchronize();
        }

        Ok(())
    }

    /// Input gradient Matmul on GPU: dX += dY * W^T
    /// dY is [m, k], W is [n, k] (W^T is [k, n]), dX is [m, n]
    pub fn matmul_a_bt(&self, dy: &[f32], w: &[f32], dx: &mut [f32], m: usize, k: usize, n: usize) -> Result<()> {
        let dy_bytes = m * k * std::mem::size_of::<f32>();
        let w_bytes = n * k * std::mem::size_of::<f32>();
        let dx_bytes = m * n * std::mem::size_of::<f32>();

        if dy_bytes > self.slab_a_size || w_bytes > self.slab_b_size || dx_bytes > self.slab_c_size {
            bail!("Matrix dimensions exceed preallocated GPU workspace slabs");
        }

        let ptr_dy = self.dev_ptr(self.slab_a_offset);
        let ptr_w = self.dev_ptr(self.slab_b_offset);
        let ptr_dx = self.dev_ptr(self.slab_c_offset);

        unsafe {
            cudaMemcpy(ptr_dy as *mut c_void, dy.as_ptr() as *const c_void, dy_bytes, CUDA_MEMCPY_HOST_TO_DEVICE);
            cudaMemcpy(ptr_w as *mut c_void, w.as_ptr() as *const c_void, w_bytes, CUDA_MEMCPY_HOST_TO_DEVICE);

            let alpha = 1.0f32;
            let beta = 0.0f32;

            // cuBLAS col-major GEMM: dX^T = W * dY^T
            let status = cublasSgemm_v2(
                self.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                ptr_w,
                k as i32,
                ptr_dy,
                k as i32,
                &beta,
                ptr_dx,
                n as i32,
            );

            if status != 0 {
                bail!("cublasSgemm_v2 (a_bt) failed with status {}", status);
            }

            cudaMemcpy(dx.as_mut_ptr() as *mut c_void, ptr_dx as *const c_void, dx_bytes, CUDA_MEMCPY_DEVICE_TO_HOST);
            cudaDeviceSynchronize();
        }

        Ok(())
    }
}

impl Drop for GpuWorkspace {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                cublasDestroy_v2(self.handle);
            }
            if !self.dev_base.is_null() {
                cudaFree(self.dev_base);
            }
        }
    }
}

/// Global Singleton GPU Workspace
static GLOBAL_GPU_WORKSPACE: Lazy<Option<Mutex<GpuWorkspace>>> = Lazy::new(|| {
    match GpuWorkspace::new() {
        Ok(ws) => {
            println!("⚡ [CUDA] Ініціалізовано 64 МБ VRAM Workspace + cuBLAS (TF32 Tensor Cores @ GA106)");
            Some(Mutex::new(ws))
        }
        Err(e) => {
            eprintln!("⚠️ [CUDA] Помилка ініціалізації GPU Workspace: {}. Використовується CPU fallback.", e);
            None
        }
    }
});

pub fn get_gpu_workspace() -> Option<&'static Mutex<GpuWorkspace>> {
    GLOBAL_GPU_WORKSPACE.as_ref()
}

pub fn matmul_gpu(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) -> Result<()> {
    if let Some(ws_mutex) = get_gpu_workspace() {
        let ws = ws_mutex.lock().unwrap();
        ws.matmul(a, b, c, m, k, n)
    } else {
        bail!("GPU workspace unavailable");
    }
}

pub fn matmul_at_b_gpu(a: &[f32], b: &[f32], dw: &mut [f32], m: usize, k: usize, n: usize) -> Result<()> {
    if let Some(ws_mutex) = get_gpu_workspace() {
        let ws = ws_mutex.lock().unwrap();
        ws.matmul_at_b(a, b, dw, m, k, n)
    } else {
        bail!("GPU workspace unavailable");
    }
}

pub fn matmul_a_bt_gpu(dy: &[f32], w: &[f32], dx: &mut [f32], m: usize, k: usize, n: usize) -> Result<()> {
    if let Some(ws_mutex) = get_gpu_workspace() {
        let ws = ws_mutex.lock().unwrap();
        ws.matmul_a_bt(dy, w, dx, m, k, n)
    } else {
        bail!("GPU workspace unavailable");
    }
}
