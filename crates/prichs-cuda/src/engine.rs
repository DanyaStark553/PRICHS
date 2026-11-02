use anyhow::{bail, Result};
use once_cell::sync::Lazy;
use std::ffi::c_void;
use std::sync::Mutex;

use crate::cublas::{
    CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_TF32_TENSOR_OP_MATH,
    CUDA_MEMCPY_DEVICE_TO_HOST, CUDA_MEMCPY_HOST_TO_DEVICE,
};

#[link(name = "cudart")]
extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: i32, stream: *mut c_void) -> i32;
    fn cudaMemset(dev_ptr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaMemsetAsync(dev_ptr: *mut c_void, value: i32, count: usize, stream: *mut c_void) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
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
    fn cublasSgemmStridedBatched(
        handle: *mut c_void,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const f32,
        lda: i32,
        stride_a: i64,
        b: *const f32,
        ldb: i32,
        stride_b: i64,
        beta: *const f32,
        c: *mut f32,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) -> i32;
    fn cublasSaxpy_v2(
        handle: *mut c_void,
        n: i32,
        alpha: *const f32,
        x: *const f32,
        incx: i32,
        y: *mut f32,
        incy: i32,
    ) -> i32;
    fn cublasSnrm2_v2(
        handle: *mut c_void,
        n: i32,
        x: *const f32,
        incx: i32,
        result: *mut f32,
    ) -> i32;
    fn cublasSscal_v2(
        handle: *mut c_void,
        n: i32,
        alpha: *const f32,
        x: *mut f32,
        incx: i32,
    ) -> i32;
}

extern "C" {
    fn launch_adamw_gpu(
        dev_weights: *mut f32,
        dev_grads: *mut f32,
        dev_exp_avg: *mut f32,
        dev_exp_avg_sq: *mut f32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: i32,
        total_elements: usize,
        stream: *mut c_void,
    );
    fn launch_accumulate_wte_grad(
        dev_grads: *mut f32,
        dev_tokens: *const u16,
        dev_dx: *const f32,
        seq_len: usize,
        hidden_dim: usize,
        stream: *mut c_void,
    );
    fn launch_fused_cross_entropy(
        dev_logits: *const f32,
        dev_targets: *const u16,
        dev_mask: *const f32,
        dev_d_logits: *mut f32,
        dev_loss_out: *mut f32,
        seq_len: usize,
        vocab_size: usize,
        active_count: f32,
        grad_scale: f32,
        stream: *mut c_void,
    );
    fn launch_rmsnorm_forward(
        dev_x: *const f32,
        dev_weight: *const f32,
        dev_out: *mut f32,
        dev_rstd_out: *mut f32,
        seq_len: usize,
        hidden_dim: usize,
        eps: f32,
        stream: *mut c_void,
    );
    fn launch_rmsnorm_backward(
        dev_dout: *const f32,
        dev_x: *const f32,
        dev_weight: *const f32,
        dev_rstd_in: *const f32,
        dev_dx: *mut f32,
        dev_dweight_accum: *mut f32,
        seq_len: usize,
        hidden_dim: usize,
        stream: *mut c_void,
    );
    fn launch_swiglu_forward(
        dev_gate: *const f32,
        dev_up: *const f32,
        dev_out: *mut f32,
        total_elements: usize,
        stream: *mut c_void,
    );
    fn launch_swiglu_backward(
        dev_dact: *const f32,
        dev_gate: *const f32,
        dev_up: *const f32,
        dev_dgate: *mut f32,
        dev_dup: *mut f32,
        total_elements: usize,
        stream: *mut c_void,
    );
    fn launch_rope(
        dev_q_or_k: *mut f32,
        dev_cos_cache: *const f32,
        dev_sin_cache: *const f32,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        backward_sign: i32,
        stream: *mut c_void,
    );
    fn launch_causal_softmax_forward(
        dev_scores: *mut f32,
        scale: f32,
        num_heads: usize,
        seq_len: usize,
        stream: *mut c_void,
    );
    fn launch_causal_softmax_backward(
        dev_d_attn: *const f32,
        dev_probs: *const f32,
        dev_d_scores: *mut f32,
        scale: f32,
        num_heads: usize,
        seq_len: usize,
        stream: *mut c_void,
    );
    fn launch_transpose_0213(
        dev_in: *const f32,
        dev_out: *mut f32,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        stream: *mut c_void,
    );
    fn launch_transpose_102_to_012(
        dev_in: *const f32,
        dev_out: *mut f32,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        stream: *mut c_void,
    );
    fn launch_repeat_kv_heads(
        dev_in: *const f32,
        dev_out: *mut f32,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        stream: *mut c_void,
    );
    fn launch_reduce_kv_heads_grad(
        dev_in: *const f32,
        dev_out: *mut f32,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        stream: *mut c_void,
    );
    fn launch_vector_add(
        dev_a: *const f32,
        dev_b: *mut f32,
        total: usize,
        stream: *mut c_void,
    );
    fn launch_moe_gather(
        dev_in: *const f32,
        dev_token_indices: *const i32,
        dev_out: *mut f32,
        n_tokens: usize,
        hidden_dim: usize,
        stream: *mut c_void,
    );
    fn launch_moe_scatter_add(
        dev_exp_out: *const f32,
        dev_token_indices: *const i32,
        dev_weights: *const f32,
        dev_accum_out: *mut f32,
        n_tokens: usize,
        hidden_dim: usize,
        stream: *mut c_void,
    );
    fn launch_moe_backward_gather_weighted(
        dev_d_moe: *const f32,
        dev_token_indices: *const i32,
        dev_weights: *const f32,
        dev_d_y_e: *mut f32,
        n_tokens: usize,
        hidden_dim: usize,
        stream: *mut c_void,
    );
    fn launch_moe_router_dot(
        dev_exp_out: *const f32,
        dev_d_moe: *const f32,
        dev_token_indices: *const i32,
        dev_scalars_out: *mut f32,
        n_tokens: usize,
        hidden_dim: usize,
        stream: *mut c_void,
    );
    fn launch_moe_scatter_add_grad(
        dev_d_x_e: *const f32,
        dev_d_x_e_tmp: *const f32,
        dev_token_indices: *const i32,
        dev_weights: *const f32,
        dev_d_normed2: *mut f32,
        n_tokens: usize,
        hidden_dim: usize,
        stream: *mut c_void,
    );
}

#[derive(Debug, Clone, Copy)]
pub struct ParamMetadata {
    pub id: usize,
    pub offset: usize,
    pub len: usize,
    pub rows: usize,
    pub cols: usize,
    pub apply_weight_decay: bool,
}

/// Persistent GPU VRAM Engine: Zero PCIe Weight Copy Architecture
pub struct GpuModelEngine {
    cublas_handle: *mut c_void,
    dev_weights: *mut f32,
    dev_grads: *mut f32,
    dev_adamw_m: *mut f32,
    dev_adamw_v: *mut f32,
    total_params: usize,
    params_meta: Vec<ParamMetadata>,

    dev_act_in: *mut f32,
    act_in_capacity: usize,
    dev_act_out: *mut f32,
    act_out_capacity: usize,
    dev_act_aux: *mut f32,
    act_aux_capacity: usize,

    // Attention Scratchpad in VRAM
    dev_q_raw: *mut f32,
    dev_k_raw: *mut f32,
    dev_v_raw: *mut f32,
    dev_q_perm: *mut f32,
    dev_k_perm: *mut f32,
    dev_v_perm: *mut f32,
    dev_scores: *mut f32,
    dev_ctx_perm: *mut f32,
    dev_ctx_raw: *mut f32,

    dev_d_ctx_raw: *mut f32,
    dev_d_ctx_perm: *mut f32,
    dev_d_scores: *mut f32,
    dev_d_scores_raw: *mut f32,
    dev_d_q_perm: *mut f32,
    dev_d_k_perm: *mut f32,
    dev_d_v_perm: *mut f32,
    dev_d_q_raw: *mut f32,
    dev_d_k_raw: *mut f32,
    dev_d_v_raw: *mut f32,
    attn_vec_capacity: usize,
    attn_scores_capacity: usize,

    // RoPE and CrossEntropy preallocated GPU buffers
    dev_cos_cache: *mut f32,
    dev_sin_cache: *mut f32,
    rope_cache_capacity: usize,

    dev_loss: *mut f32,
    dev_targets: *mut u16,
    dev_mask: *mut f32,
    target_mask_capacity: usize,

    // MoE Scratchpad in VRAM (Zero-CPU activations)
    dev_moe_out: *mut f32,
    dev_shared_gate: *mut f32,
    dev_shared_up: *mut f32,
    dev_shared_act: *mut f32,
    dev_shared_out: *mut f32,
    dev_exp_x: *mut f32,
    dev_exp_gate: *mut f32,
    dev_exp_up: *mut f32,
    dev_exp_act: *mut f32,
    dev_exp_out: *mut f32,
    dev_exp_dact: *mut f32,
    dev_exp_dgate: *mut f32,
    dev_exp_dup: *mut f32,
    dev_exp_dx: *mut f32,
    dev_exp_dx_tmp: *mut f32,
    dev_exp_dy: *mut f32,
    dev_exp_indices: *mut i32,
    dev_exp_weights: *mut f32,
    dev_exp_scalars: *mut f32,
    dev_rstd: *mut f32,
    moe_dim_capacity: usize,
}

unsafe impl Send for GpuModelEngine {}
unsafe impl Sync for GpuModelEngine {}

impl GpuModelEngine {
    pub fn new(param_shapes: &[(usize, usize, bool)]) -> Result<Self> {
        let mut cublas_handle: *mut c_void = std::ptr::null_mut();
        let status = unsafe { cublasCreate_v2(&mut cublas_handle) };
        if status != 0 || cublas_handle.is_null() {
            bail!("cublasCreate_v2 failed with status {}", status);
        }

        unsafe { cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH) };

        let mut offset = 0usize;
        let mut params_meta = Vec::with_capacity(param_shapes.len());
        for (id, &(rows, cols, apply_wd)) in param_shapes.iter().enumerate() {
            let len = rows * cols;
            params_meta.push(ParamMetadata {
                id,
                offset,
                len,
                rows,
                cols,
                apply_weight_decay: apply_wd,
            });
            offset += len;
        }

        let total_params = offset;
        let bytes_per_buffer = total_params * std::mem::size_of::<f32>();

        let mut dev_weights: *mut c_void = std::ptr::null_mut();
        let mut dev_grads: *mut c_void = std::ptr::null_mut();
        let mut dev_adamw_m: *mut c_void = std::ptr::null_mut();
        let mut dev_adamw_v: *mut c_void = std::ptr::null_mut();

        unsafe {
            if cudaMalloc(&mut dev_weights, bytes_per_buffer) != 0 || dev_weights.is_null() {
                bail!("cudaMalloc failed for dev_weights ({:.2} MB)", bytes_per_buffer as f64 / 1e6);
            }
            if cudaMalloc(&mut dev_grads, bytes_per_buffer) != 0 || dev_grads.is_null() {
                bail!("cudaMalloc failed for dev_grads ({:.2} MB)", bytes_per_buffer as f64 / 1e6);
            }
            if cudaMalloc(&mut dev_adamw_m, bytes_per_buffer) != 0 || dev_adamw_m.is_null() {
                bail!("cudaMalloc failed for dev_adamw_m ({:.2} MB)", bytes_per_buffer as f64 / 1e6);
            }
            if cudaMalloc(&mut dev_adamw_v, bytes_per_buffer) != 0 || dev_adamw_v.is_null() {
                bail!("cudaMalloc failed for dev_adamw_v ({:.2} MB)", bytes_per_buffer as f64 / 1e6);
            }

            cudaMemset(dev_grads, 0, bytes_per_buffer);
            cudaMemset(dev_adamw_m, 0, bytes_per_buffer);
            cudaMemset(dev_adamw_v, 0, bytes_per_buffer);
        }

        let act_capacity = 64 * 1024 * 1024; // 256 MB per activation buffer
        let act_bytes = act_capacity * std::mem::size_of::<f32>();

        let mut dev_act_in: *mut c_void = std::ptr::null_mut();
        let mut dev_act_out: *mut c_void = std::ptr::null_mut();
        let mut dev_act_aux: *mut c_void = std::ptr::null_mut();

        let attn_vec_capacity = 4 * 1024 * 1024; // 16 MB per vector scratchpad
        let attn_vec_bytes = attn_vec_capacity * std::mem::size_of::<f32>();
        let attn_scores_capacity = 32 * 1024 * 1024; // 128 MB per scores scratchpad
        let attn_scores_bytes = attn_scores_capacity * std::mem::size_of::<f32>();
        let rope_cache_capacity = 128 * 1024;
        let rope_cache_bytes = rope_cache_capacity * std::mem::size_of::<f32>();
        let target_mask_capacity = 4096usize;

        let mut dev_q_raw: *mut c_void = std::ptr::null_mut();
        let mut dev_k_raw: *mut c_void = std::ptr::null_mut();
        let mut dev_v_raw: *mut c_void = std::ptr::null_mut();
        let mut dev_q_perm: *mut c_void = std::ptr::null_mut();
        let mut dev_k_perm: *mut c_void = std::ptr::null_mut();
        let mut dev_v_perm: *mut c_void = std::ptr::null_mut();
        let mut dev_scores: *mut c_void = std::ptr::null_mut();
        let mut dev_ctx_perm: *mut c_void = std::ptr::null_mut();
        let mut dev_ctx_raw: *mut c_void = std::ptr::null_mut();

        let mut dev_d_ctx_raw: *mut c_void = std::ptr::null_mut();
        let mut dev_d_ctx_perm: *mut c_void = std::ptr::null_mut();
        let mut dev_d_scores: *mut c_void = std::ptr::null_mut();
        let mut dev_d_scores_raw: *mut c_void = std::ptr::null_mut();
        let mut dev_d_q_perm: *mut c_void = std::ptr::null_mut();
        let mut dev_d_k_perm: *mut c_void = std::ptr::null_mut();
        let mut dev_d_v_perm: *mut c_void = std::ptr::null_mut();
        let mut dev_d_q_raw: *mut c_void = std::ptr::null_mut();
        let mut dev_d_k_raw: *mut c_void = std::ptr::null_mut();
        let mut dev_d_v_raw: *mut c_void = std::ptr::null_mut();

        let mut dev_cos_cache: *mut c_void = std::ptr::null_mut();
        let mut dev_sin_cache: *mut c_void = std::ptr::null_mut();
        let mut dev_loss: *mut c_void = std::ptr::null_mut();
        let mut dev_targets: *mut c_void = std::ptr::null_mut();
        let mut dev_mask: *mut c_void = std::ptr::null_mut();

        let moe_dim_capacity = 8 * 1024 * 1024;
        let moe_dim_bytes = moe_dim_capacity * std::mem::size_of::<f32>();
        let moe_hidden_capacity = 4 * 1024 * 1024;
        let moe_hidden_bytes = moe_hidden_capacity * std::mem::size_of::<f32>();
        let moe_meta_capacity = 16384usize;

        let mut dev_moe_out: *mut c_void = std::ptr::null_mut();
        let mut dev_shared_gate: *mut c_void = std::ptr::null_mut();
        let mut dev_shared_up: *mut c_void = std::ptr::null_mut();
        let mut dev_shared_act: *mut c_void = std::ptr::null_mut();
        let mut dev_shared_out: *mut c_void = std::ptr::null_mut();

        let mut dev_exp_x: *mut c_void = std::ptr::null_mut();
        let mut dev_exp_gate: *mut c_void = std::ptr::null_mut();
        let mut dev_exp_up: *mut c_void = std::ptr::null_mut();
        let mut dev_exp_act: *mut c_void = std::ptr::null_mut();
        let mut dev_exp_out: *mut c_void = std::ptr::null_mut();
        let mut dev_exp_dact: *mut c_void = std::ptr::null_mut();
        let mut dev_exp_dgate: *mut c_void = std::ptr::null_mut();
        let mut dev_exp_dup: *mut c_void = std::ptr::null_mut();
        let mut dev_exp_dx: *mut c_void = std::ptr::null_mut();
        let mut dev_exp_dx_tmp: *mut c_void = std::ptr::null_mut();
        let mut dev_exp_dy: *mut c_void = std::ptr::null_mut();

        let mut dev_exp_indices: *mut c_void = std::ptr::null_mut();
        let mut dev_exp_weights: *mut c_void = std::ptr::null_mut();
        let mut dev_exp_scalars: *mut c_void = std::ptr::null_mut();
        let mut dev_rstd: *mut c_void = std::ptr::null_mut();

        unsafe {
            if cudaMalloc(&mut dev_act_in, act_bytes) != 0 { bail!("cudaMalloc failed for dev_act_in"); }
            if cudaMalloc(&mut dev_act_out, act_bytes) != 0 { bail!("cudaMalloc failed for dev_act_out"); }
            if cudaMalloc(&mut dev_act_aux, act_bytes) != 0 { bail!("cudaMalloc failed for dev_act_aux"); }

            let alloc_vec = |ptr: &mut *mut c_void| -> Result<()> {
                if cudaMalloc(ptr, attn_vec_bytes) != 0 || ptr.is_null() {
                    bail!("cudaMalloc failed for attention vector buffer");
                }
                Ok(())
            };
            let alloc_scores = |ptr: &mut *mut c_void| -> Result<()> {
                if cudaMalloc(ptr, attn_scores_bytes) != 0 || ptr.is_null() {
                    bail!("cudaMalloc failed for attention scores buffer");
                }
                Ok(())
            };

            alloc_vec(&mut dev_q_raw)?;
            alloc_vec(&mut dev_k_raw)?;
            alloc_vec(&mut dev_v_raw)?;
            alloc_vec(&mut dev_q_perm)?;
            alloc_vec(&mut dev_k_perm)?;
            alloc_vec(&mut dev_v_perm)?;
            alloc_vec(&mut dev_ctx_perm)?;
            alloc_vec(&mut dev_ctx_raw)?;
            alloc_vec(&mut dev_d_ctx_raw)?;
            alloc_vec(&mut dev_d_ctx_perm)?;
            alloc_vec(&mut dev_d_q_perm)?;
            alloc_vec(&mut dev_d_k_perm)?;
            alloc_vec(&mut dev_d_v_perm)?;
            alloc_vec(&mut dev_d_q_raw)?;
            alloc_vec(&mut dev_d_k_raw)?;
            alloc_vec(&mut dev_d_v_raw)?;

            alloc_scores(&mut dev_scores)?;
            alloc_scores(&mut dev_d_scores)?;
            alloc_scores(&mut dev_d_scores_raw)?;

            if cudaMalloc(&mut dev_cos_cache, rope_cache_bytes) != 0 { bail!("cudaMalloc dev_cos_cache failed"); }
            if cudaMalloc(&mut dev_sin_cache, rope_cache_bytes) != 0 { bail!("cudaMalloc dev_sin_cache failed"); }
            if cudaMalloc(&mut dev_loss, std::mem::size_of::<f32>()) != 0 { bail!("cudaMalloc dev_loss failed"); }
            if cudaMalloc(&mut dev_targets, target_mask_capacity * std::mem::size_of::<u16>()) != 0 { bail!("cudaMalloc dev_targets failed"); }
            if cudaMalloc(&mut dev_mask, target_mask_capacity * std::mem::size_of::<f32>()) != 0 { bail!("cudaMalloc dev_mask failed"); }

            let alloc_moe_dim = |ptr: &mut *mut c_void| -> Result<()> {
                if cudaMalloc(ptr, moe_dim_bytes) != 0 || ptr.is_null() {
                    bail!("cudaMalloc failed for MoE dim buffer");
                }
                Ok(())
            };
            let alloc_moe_hidden = |ptr: &mut *mut c_void| -> Result<()> {
                if cudaMalloc(ptr, moe_hidden_bytes) != 0 || ptr.is_null() {
                    bail!("cudaMalloc failed for MoE hidden buffer");
                }
                Ok(())
            };

            alloc_moe_hidden(&mut dev_moe_out)?;
            alloc_moe_dim(&mut dev_shared_gate)?;
            alloc_moe_dim(&mut dev_shared_up)?;
            alloc_moe_dim(&mut dev_shared_act)?;
            alloc_moe_hidden(&mut dev_shared_out)?;

            alloc_moe_hidden(&mut dev_exp_x)?;
            alloc_moe_dim(&mut dev_exp_gate)?;
            alloc_moe_dim(&mut dev_exp_up)?;
            alloc_moe_dim(&mut dev_exp_act)?;
            alloc_moe_hidden(&mut dev_exp_out)?;
            alloc_moe_dim(&mut dev_exp_dact)?;
            alloc_moe_dim(&mut dev_exp_dgate)?;
            alloc_moe_dim(&mut dev_exp_dup)?;
            alloc_moe_hidden(&mut dev_exp_dx)?;
            alloc_moe_hidden(&mut dev_exp_dx_tmp)?;
            alloc_moe_hidden(&mut dev_exp_dy)?;

            if cudaMalloc(&mut dev_exp_indices, moe_meta_capacity * std::mem::size_of::<i32>()) != 0 {
                bail!("cudaMalloc failed for dev_exp_indices");
            }
            if cudaMalloc(&mut dev_exp_weights, moe_meta_capacity * std::mem::size_of::<f32>()) != 0 {
                bail!("cudaMalloc failed for dev_exp_weights");
            }
            if cudaMalloc(&mut dev_exp_scalars, moe_meta_capacity * std::mem::size_of::<f32>()) != 0 {
                bail!("cudaMalloc failed for dev_exp_scalars");
            }
            if cudaMalloc(&mut dev_rstd, moe_meta_capacity * std::mem::size_of::<f32>()) != 0 {
                bail!("cudaMalloc failed for dev_rstd");
            }
        }

        let scratchpad_bytes = attn_vec_bytes * 16 + attn_scores_bytes * 3 + rope_cache_bytes * 2;
        let total_vram_mb = (bytes_per_buffer * 4 + act_bytes * 3 + scratchpad_bytes) as f64 / (1024.0 * 1024.0);
        println!("🚀 [CUDA VRAM Engine] Виділено {:.2} MB ({:.2} GB) постійної VRAM на RTX 3060 (включно зі скретчпадами Уваги, MoE та Cross-Entropy)",
            total_vram_mb, total_vram_mb / 1024.0);

        Ok(Self {
            cublas_handle,
            dev_weights: dev_weights as *mut f32,
            dev_grads: dev_grads as *mut f32,
            dev_adamw_m: dev_adamw_m as *mut f32,
            dev_adamw_v: dev_adamw_v as *mut f32,
            total_params,
            params_meta,
            dev_act_in: dev_act_in as *mut f32,
            act_in_capacity: act_capacity,
            dev_act_out: dev_act_out as *mut f32,
            act_out_capacity: act_capacity,
            dev_act_aux: dev_act_aux as *mut f32,
            act_aux_capacity: act_capacity,

            dev_q_raw: dev_q_raw as *mut f32,
            dev_k_raw: dev_k_raw as *mut f32,
            dev_v_raw: dev_v_raw as *mut f32,
            dev_q_perm: dev_q_perm as *mut f32,
            dev_k_perm: dev_k_perm as *mut f32,
            dev_v_perm: dev_v_perm as *mut f32,
            dev_scores: dev_scores as *mut f32,
            dev_ctx_perm: dev_ctx_perm as *mut f32,
            dev_ctx_raw: dev_ctx_raw as *mut f32,

            dev_d_ctx_raw: dev_d_ctx_raw as *mut f32,
            dev_d_ctx_perm: dev_d_ctx_perm as *mut f32,
            dev_d_scores: dev_d_scores as *mut f32,
            dev_d_scores_raw: dev_d_scores_raw as *mut f32,
            dev_d_q_perm: dev_d_q_perm as *mut f32,
            dev_d_k_perm: dev_d_k_perm as *mut f32,
            dev_d_v_perm: dev_d_v_perm as *mut f32,
            dev_d_q_raw: dev_d_q_raw as *mut f32,
            dev_d_k_raw: dev_d_k_raw as *mut f32,
            dev_d_v_raw: dev_d_v_raw as *mut f32,
            attn_vec_capacity,
            attn_scores_capacity,

            dev_cos_cache: dev_cos_cache as *mut f32,
            dev_sin_cache: dev_sin_cache as *mut f32,
            rope_cache_capacity,

            dev_loss: dev_loss as *mut f32,
            dev_targets: dev_targets as *mut u16,
            dev_mask: dev_mask as *mut f32,
            target_mask_capacity,

            dev_moe_out: dev_moe_out as *mut f32,
            dev_shared_gate: dev_shared_gate as *mut f32,
            dev_shared_up: dev_shared_up as *mut f32,
            dev_shared_act: dev_shared_act as *mut f32,
            dev_shared_out: dev_shared_out as *mut f32,
            dev_exp_x: dev_exp_x as *mut f32,
            dev_exp_gate: dev_exp_gate as *mut f32,
            dev_exp_up: dev_exp_up as *mut f32,
            dev_exp_act: dev_exp_act as *mut f32,
            dev_exp_out: dev_exp_out as *mut f32,
            dev_exp_dact: dev_exp_dact as *mut f32,
            dev_exp_dgate: dev_exp_dgate as *mut f32,
            dev_exp_dup: dev_exp_dup as *mut f32,
            dev_exp_dx: dev_exp_dx as *mut f32,
            dev_exp_dx_tmp: dev_exp_dx_tmp as *mut f32,
            dev_exp_dy: dev_exp_dy as *mut f32,
            dev_exp_indices: dev_exp_indices as *mut i32,
            dev_exp_weights: dev_exp_weights as *mut f32,
            dev_exp_scalars: dev_exp_scalars as *mut f32,
            dev_rstd: dev_rstd as *mut f32,
            moe_dim_capacity,
        })
    }

    pub fn total_params(&self) -> usize {
        self.total_params
    }

    /// Query current free and total GPU VRAM directly via CUDA Runtime API (cudaMemGetInfo)
    pub fn get_gpu_memory_info() -> Result<(usize, usize)> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        unsafe {
            let status = cudaMemGetInfo(&mut free, &mut total);
            if status != 0 {
                bail!("cudaMemGetInfo failed with status {}", status);
            }
        }
        Ok((free, total))
    }

    pub fn upload_weights(&mut self, param_slices: &[&[f32]]) -> Result<()> {
        if param_slices.len() != self.params_meta.len() {
            bail!("param_slices count {} != meta count {}", param_slices.len(), self.params_meta.len());
        }

        for (meta, &host_slice) in self.params_meta.iter().zip(param_slices.iter()) {
            if host_slice.len() != meta.len {
                bail!("Param {} length mismatch: host {} != meta {}", meta.id, host_slice.len(), meta.len);
            }
            unsafe {
                let dst = self.dev_weights.add(meta.offset);
                let bytes = meta.len * std::mem::size_of::<f32>();
                let status = cudaMemcpy(dst as *mut c_void, host_slice.as_ptr() as *const c_void, bytes, CUDA_MEMCPY_HOST_TO_DEVICE);
                if status != 0 {
                    bail!("cudaMemcpy failed to upload param {}", meta.id);
                }
            }
        }
        unsafe { cudaDeviceSynchronize(); }
        println!("💾 [CUDA VRAM Engine] Всі ваги моделі успішно завантажено в VRAM (Zero PCIe Copy активовано)!");
        Ok(())
    }

    pub fn download_weights(&self, param_slices: &mut [&mut [f32]]) -> Result<()> {
        if param_slices.len() != self.params_meta.len() {
            bail!("param_slices count {} != meta count {}", param_slices.len(), self.params_meta.len());
        }

        for (meta, host_slice) in self.params_meta.iter().zip(param_slices.iter_mut()) {
            unsafe {
                let src = self.dev_weights.add(meta.offset);
                let bytes = meta.len * std::mem::size_of::<f32>();
                let status = cudaMemcpy(host_slice.as_mut_ptr() as *mut c_void, src as *const c_void, bytes, CUDA_MEMCPY_DEVICE_TO_HOST);
                if status != 0 {
                    bail!("cudaMemcpy failed to download param {}", meta.id);
                }
            }
        }
        unsafe { cudaDeviceSynchronize(); }
        Ok(())
    }

    pub fn zero_grads(&mut self) -> Result<()> {
        let bytes = self.total_params * std::mem::size_of::<f32>();
        unsafe {
            let status = cudaMemsetAsync(self.dev_grads as *mut c_void, 0, bytes, std::ptr::null_mut());
            if status != 0 {
                bail!("cudaMemsetAsync failed: status {}", status);
            }
        }
        Ok(())
    }

    /// Hardware-accelerated gradient clipping in VRAM via cuBLAS Snrm2 and Sscal
    pub fn clip_grad_norm(&mut self, max_norm: f32) -> Result<f32> {
        if max_norm <= 0.0 {
            return Ok(0.0);
        }
        let mut norm: f32 = 0.0;
        unsafe {
            let status = cublasSnrm2_v2(
                self.cublas_handle,
                self.total_params as i32,
                self.dev_grads,
                1,
                &mut norm,
            );
            if status != 0 {
                bail!("cublasSnrm2_v2 failed with status {}", status);
            }
            if norm > max_norm && norm.is_finite() {
                let scale = max_norm / norm;
                let status_scale = cublasSscal_v2(
                    self.cublas_handle,
                    self.total_params as i32,
                    &scale,
                    self.dev_grads,
                    1,
                );
                if status_scale != 0 {
                    bail!("cublasSscal_v2 failed with status {}", status_scale);
                }
            }
        }
        Ok(norm)
    }

    /// Forward Matmul: Out = X * W
    /// X is [m, k], W (param_id) is [k, n], Out is [m, n]
    pub fn forward_gemm(
        &self,
        param_id: usize,
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        let meta = &self.params_meta[param_id];
        if meta.rows != k || meta.cols != n {
            bail!("Param {} shape mismatch: expected [{}, {}], meta is [{}, {}]",
                param_id, k, n, meta.rows, meta.cols);
        }

        let x_bytes = m * k * std::mem::size_of::<f32>();
        let out_bytes = m * n * std::mem::size_of::<f32>();

        if m * k > self.act_in_capacity || m * n > self.act_out_capacity {
            bail!("Activation dimensions exceed preallocated VRAM workspace");
        }

        unsafe {
            cudaMemcpyAsync(
                self.dev_act_in as *mut c_void,
                x.as_ptr() as *const c_void,
                x_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );

            let w_ptr = self.dev_weights.add(meta.offset);
            let alpha = 1.0f32;
            let beta = 0.0f32;

            // cuBLAS col-major: Out^T = W^T * X^T
            let status = cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                w_ptr,
                n as i32,
                self.dev_act_in,
                k as i32,
                &beta,
                self.dev_act_out,
                n as i32,
            );

            if status != 0 {
                bail!("cublasSgemm_v2 forward failed with status {}", status);
            }

            cudaMemcpyAsync(
                out.as_mut_ptr() as *mut c_void,
                self.dev_act_out as *const c_void,
                out_bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );

            cudaDeviceSynchronize();
        }

        Ok(())
    }

    /// Forward Transposed Matmul: Out = X * W^T
    /// X is [m, k], W (param_id) is [n, k] (so W^T is [k, n]), Out is [m, n]
    /// Used for LM Head Projection (tied embeddings) directly in VRAM
    pub fn forward_transposed_gemm(
        &self,
        param_id: usize,
        x: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        let meta = &self.params_meta[param_id];
        if meta.rows != n || meta.cols != k {
            bail!("Param {} shape mismatch for forward_transposed: expected [{}, {}], meta is [{}, {}]",
                param_id, n, k, meta.rows, meta.cols);
        }

        let x_bytes = m * k * std::mem::size_of::<f32>();
        let out_bytes = m * n * std::mem::size_of::<f32>();

        if m * k > self.act_in_capacity || m * n > self.act_out_capacity {
            bail!("Activation dimensions exceed preallocated VRAM workspace");
        }

        unsafe {
            cudaMemcpyAsync(
                self.dev_act_in as *mut c_void,
                x.as_ptr() as *const c_void,
                x_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );

            let w_ptr = self.dev_weights.add(meta.offset);
            let alpha = 1.0f32;
            let beta = 0.0f32;

            // cuBLAS col-major: Out^T = W * X^T
            // W is [n, k] row-major -> [k, n] col-major, so we do OP_T to get [n, k] with ld = k!
            let status = cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                w_ptr,
                k as i32,
                self.dev_act_in,
                k as i32,
                &beta,
                self.dev_act_out,
                n as i32,
            );

            if status != 0 {
                bail!("cublasSgemm_v2 forward_transposed failed with status {}", status);
            }

            cudaMemcpyAsync(
                out.as_mut_ptr() as *mut c_void,
                self.dev_act_out as *const c_void,
                out_bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );

            cudaDeviceSynchronize();
        }

        Ok(())
    }

    /// Backward Data Matmul: dX = dY * W^T
    /// dY is [m, n], W (param_id) is [k, n] (W^T is [n, k]), dX is [m, k]
    pub fn backward_data_gemm(
        &self,
        param_id: usize,
        dy: &[f32],
        dx: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        let meta = &self.params_meta[param_id];
        if meta.rows != k || meta.cols != n {
            bail!("Param {} shape mismatch in backward_data_gemm: expected [{}, {}], meta is [{}, {}]",
                param_id, k, n, meta.rows, meta.cols);
        }
        let dy_bytes = m * n * std::mem::size_of::<f32>();
        let dx_bytes = m * k * std::mem::size_of::<f32>();

        if m * n > self.act_in_capacity || m * k > self.act_out_capacity {
            bail!("Activation dimensions exceed preallocated VRAM workspace");
        }

        unsafe {
            cudaMemcpyAsync(
                self.dev_act_in as *mut c_void,
                dy.as_ptr() as *const c_void,
                dy_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );

            let w_ptr = self.dev_weights.add(meta.offset);
            let alpha = 1.0f32;
            let beta = 0.0f32;

            // cuBLAS col-major: dX^T = W * dY^T
            let status = cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                k as i32,
                m as i32,
                n as i32,
                &alpha,
                w_ptr,
                n as i32,
                self.dev_act_in,
                n as i32,
                &beta,
                self.dev_act_out,
                k as i32,
            );

            if status != 0 {
                bail!("cublasSgemm_v2 backward_data failed with status {}", status);
            }

            cudaMemcpyAsync(
                dx.as_mut_ptr() as *mut c_void,
                self.dev_act_out as *const c_void,
                dx_bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );

            cudaDeviceSynchronize();
        }

        Ok(())
    }

    /// Backward Weight Matmul directly into VRAM dev_grads: dW += A^T * B
    /// A is [m, k], B is [m, n], dW is [k, n]
    pub fn backward_weight_gemm(
        &self,
        param_id: usize,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        let meta = &self.params_meta[param_id];
        if meta.rows != k || meta.cols != n {
            bail!("Param {} shape mismatch in backward_weight_gemm: expected [{}, {}], meta is [{}, {}]",
                param_id, k, n, meta.rows, meta.cols);
        }
        let a_bytes = m * k * std::mem::size_of::<f32>();
        let b_bytes = m * n * std::mem::size_of::<f32>();

        if m * k > self.act_in_capacity || m * n > self.act_aux_capacity {
            bail!("Activation dimensions exceed preallocated VRAM workspace");
        }

        unsafe {
            cudaMemcpyAsync(
                self.dev_act_in as *mut c_void,
                a.as_ptr() as *const c_void,
                a_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );
            cudaMemcpyAsync(
                self.dev_act_aux as *mut c_void,
                b.as_ptr() as *const c_void,
                b_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );

            let dw_ptr = self.dev_grads.add(meta.offset);
            let alpha = 1.0f32;
            let beta = 1.0f32; // In-place accumulation!

            // cuBLAS col-major: dW^T += B^T * A
            let status = cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                n as i32,
                k as i32,
                m as i32,
                &alpha,
                self.dev_act_aux,
                n as i32,
                self.dev_act_in,
                k as i32,
                &beta,
                dw_ptr,
                n as i32,
            );

            if status != 0 {
                bail!("cublasSgemm_v2 backward_weight failed with status {}", status);
            }

            cudaDeviceSynchronize();
        }

        Ok(())
    }

    /// Accumulate 1D parameter gradient (e.g. RMSNorm weights) directly into dev_grads via cuBLAS Saxpy
    pub fn accumulate_vector_grad(&mut self, param_id: usize, grad: &[f32]) -> Result<()> {
        let meta = &self.params_meta[param_id];
        if meta.len != grad.len() {
            bail!("Param {} len mismatch: {} != {}", param_id, meta.len, grad.len());
        }
        let bytes = meta.len * std::mem::size_of::<f32>();
        if bytes > self.act_in_capacity * std::mem::size_of::<f32>() {
            bail!("Grad size exceeds dev_act_in workspace");
        }

        unsafe {
            cudaMemcpyAsync(
                self.dev_act_in as *mut c_void,
                grad.as_ptr() as *const c_void,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );
            let alpha = 1.0f32;
            let status = cublasSaxpy_v2(
                self.cublas_handle,
                meta.len as i32,
                &alpha,
                self.dev_act_in,
                1,
                self.dev_grads.add(meta.offset),
                1,
            );
            if status != 0 {
                bail!("cublasSaxpy_v2 failed with status {}", status);
            }
            cudaDeviceSynchronize();
        }
        Ok(())
    }

    /// Accumulate input token embedding gradients (WTE) directly in VRAM via atomicAdd CUDA kernel
    pub fn accumulate_token_embeddings_grad(
        &mut self,
        tokens: &[u16],
        dx: &[f32],
        hidden_dim: usize,
    ) -> Result<()> {
        let seq_len = tokens.len();
        if seq_len * hidden_dim != dx.len() {
            bail!("dx len mismatch: {} != {}", dx.len(), seq_len * hidden_dim);
        }
        let dx_bytes = dx.len() * std::mem::size_of::<f32>();
        let tokens_bytes = tokens.len() * std::mem::size_of::<u16>();

        if dx_bytes > self.act_in_capacity * std::mem::size_of::<f32>() {
            bail!("dx exceeds dev_act_in workspace");
        }
        if tokens_bytes > self.act_aux_capacity * std::mem::size_of::<f32>() {
            bail!("tokens exceeds dev_act_aux workspace");
        }

        unsafe {
            cudaMemcpyAsync(
                self.dev_act_in as *mut c_void,
                dx.as_ptr() as *const c_void,
                dx_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );
            cudaMemcpyAsync(
                self.dev_act_aux as *mut c_void,
                tokens.as_ptr() as *const c_void,
                tokens_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );
            let wte_meta = &self.params_meta[0];
            launch_accumulate_wte_grad(
                self.dev_grads.add(wte_meta.offset),
                self.dev_act_aux as *const u16,
                self.dev_act_in as *const f32,
                seq_len,
                hidden_dim,
                std::ptr::null_mut(),
            );
            cudaDeviceSynchronize();
        }
        Ok(())
    }

    pub fn adamw_step(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: usize,
    ) -> Result<()> {
        unsafe {
            launch_adamw_gpu(
                self.dev_weights,
                self.dev_grads,
                self.dev_adamw_m,
                self.dev_adamw_v,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step as i32,
                self.total_params,
                std::ptr::null_mut(),
            );
            cudaDeviceSynchronize();
        }
        Ok(())
    }

    pub fn upload_rope_cache(&mut self, cos_cache: &[f32], sin_cache: &[f32]) -> Result<()> {
        if cos_cache.len() != sin_cache.len() {
            bail!("cos_cache and sin_cache must have equal length");
        }
        if cos_cache.len() > self.rope_cache_capacity {
            bail!("RoPE cache length {} exceeds capacity {}", cos_cache.len(), self.rope_cache_capacity);
        }
        let bytes = cos_cache.len() * std::mem::size_of::<f32>();
        unsafe {
            let s1 = cudaMemcpyAsync(self.dev_cos_cache as *mut c_void, cos_cache.as_ptr() as *const c_void, bytes, CUDA_MEMCPY_HOST_TO_DEVICE, std::ptr::null_mut());
            let s2 = cudaMemcpyAsync(self.dev_sin_cache as *mut c_void, sin_cache.as_ptr() as *const c_void, bytes, CUDA_MEMCPY_HOST_TO_DEVICE, std::ptr::null_mut());
            if s1 != 0 || s2 != 0 {
                bail!("cudaMemcpyAsync failed for RoPE cache");
            }
            cudaDeviceSynchronize();
        }
        Ok(())
    }

    pub fn gpu_batched_attention_forward(
        &mut self,
        normed1: &[f32],
        q_id: usize,
        k_id: usize,
        v_id: usize,
        out_id: usize,
        attn_out: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let hidden_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        if normed1.len() != seq_len * hidden_dim {
            bail!("normed1 length mismatch: {} != {}", normed1.len(), seq_len * hidden_dim);
        }
        if attn_out.len() != seq_len * hidden_dim {
            bail!("attn_out length mismatch: {} != {}", attn_out.len(), seq_len * hidden_dim);
        }
        if seq_len * hidden_dim > self.attn_vec_capacity {
            bail!("Attention dimensions exceed attn_vec_capacity");
        }
        if num_heads * seq_len * seq_len > self.attn_scores_capacity {
            bail!("Attention scores dimensions exceed attn_scores_capacity");
        }

        let normed_bytes = normed1.len() * std::mem::size_of::<f32>();
        let q_bytes = seq_len * hidden_dim * std::mem::size_of::<f32>();
        let kv_bytes = seq_len * kv_dim * std::mem::size_of::<f32>();
        let scores_bytes = num_heads * seq_len * seq_len * std::mem::size_of::<f32>();

        unsafe {
            cudaMemcpyAsync(
                self.dev_act_in as *mut c_void,
                normed1.as_ptr() as *const c_void,
                normed_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );

            let meta_q = &self.params_meta[q_id];
            let alpha = 1.0f32;
            let beta = 0.0f32;
            let status_q = cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                hidden_dim as i32,
                seq_len as i32,
                hidden_dim as i32,
                &alpha,
                self.dev_weights.add(meta_q.offset),
                hidden_dim as i32,
                self.dev_act_in,
                hidden_dim as i32,
                &beta,
                self.dev_q_raw,
                hidden_dim as i32,
            );
            if status_q != 0 { bail!("cublasSgemm_v2 Q failed with status {}", status_q); }

            let meta_k = &self.params_meta[k_id];
            let status_k = cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                kv_dim as i32,
                seq_len as i32,
                hidden_dim as i32,
                &alpha,
                self.dev_weights.add(meta_k.offset),
                kv_dim as i32,
                self.dev_act_in,
                hidden_dim as i32,
                &beta,
                self.dev_k_raw,
                kv_dim as i32,
            );
            if status_k != 0 { bail!("cublasSgemm_v2 K failed with status {}", status_k); }

            let meta_v = &self.params_meta[v_id];
            let status_v = cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                kv_dim as i32,
                seq_len as i32,
                hidden_dim as i32,
                &alpha,
                self.dev_weights.add(meta_v.offset),
                kv_dim as i32,
                self.dev_act_in,
                hidden_dim as i32,
                &beta,
                self.dev_v_raw,
                kv_dim as i32,
            );
            if status_v != 0 { bail!("cublasSgemm_v2 V failed with status {}", status_v); }

            // RoPE Forward in-place
            launch_rope(self.dev_q_raw, self.dev_cos_cache, self.dev_sin_cache, seq_len, num_heads, head_dim, 1, std::ptr::null_mut());
            launch_rope(self.dev_k_raw, self.dev_cos_cache, self.dev_sin_cache, seq_len, num_kv_heads, head_dim, 1, std::ptr::null_mut());

            // Permute Q -> dev_q_perm [H, T, D]
            launch_transpose_0213(self.dev_q_raw, self.dev_q_perm, seq_len, num_heads, head_dim, std::ptr::null_mut());

            // Repeat KV & Permute -> dev_k_perm [H, T, D] and dev_v_perm [H, T, D]
            launch_repeat_kv_heads(self.dev_k_raw, self.dev_k_perm, seq_len, num_heads, num_kv_heads, head_dim, std::ptr::null_mut());
            launch_repeat_kv_heads(self.dev_v_raw, self.dev_v_perm, seq_len, num_heads, num_kv_heads, head_dim, std::ptr::null_mut());

            // Batched GEMM 1: Scores = Q * K^T -> dev_scores [H, T, T]
            let status_scores = cublasSgemmStridedBatched(
                self.cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                seq_len as i32,
                seq_len as i32,
                head_dim as i32,
                &alpha,
                self.dev_k_perm,
                head_dim as i32,
                (seq_len * head_dim) as i64,
                self.dev_q_perm,
                head_dim as i32,
                (seq_len * head_dim) as i64,
                &beta,
                self.dev_scores,
                seq_len as i32,
                (seq_len * seq_len) as i64,
                num_heads as i32,
            );
            if status_scores != 0 { bail!("cublasSgemmStridedBatched Scores failed with status {}", status_scores); }

            // Causal Softmax in VRAM
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            launch_causal_softmax_forward(
                self.dev_scores,
                scale,
                num_heads,
                seq_len,
                std::ptr::null_mut(),
            );

            // Batched GEMM 2: Ctx = Scores * V -> dev_ctx_perm [H, T, D]
            let status_ctx = cublasSgemmStridedBatched(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                head_dim as i32,
                seq_len as i32,
                seq_len as i32,
                &alpha,
                self.dev_v_perm,
                head_dim as i32,
                (seq_len * head_dim) as i64,
                self.dev_scores,
                seq_len as i32,
                (seq_len * seq_len) as i64,
                &beta,
                self.dev_ctx_perm,
                head_dim as i32,
                (seq_len * head_dim) as i64,
                num_heads as i32,
            );
            if status_ctx != 0 { bail!("cublasSgemmStridedBatched Ctx failed with status {}", status_ctx); }

            // Permute Ctx [H, T, D] -> dev_ctx_raw [T, H, D]
            launch_transpose_102_to_012(
                self.dev_ctx_perm,
                self.dev_ctx_raw,
                num_heads,
                seq_len,
                head_dim,
                std::ptr::null_mut(),
            );

            // Out Projection: dev_ctx_raw [seq_len, hidden_dim] * Wo [hidden_dim, hidden_dim] -> dev_act_out
            let meta_out = &self.params_meta[out_id];
            let status_out = cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                hidden_dim as i32,
                seq_len as i32,
                hidden_dim as i32,
                &alpha,
                self.dev_weights.add(meta_out.offset),
                hidden_dim as i32,
                self.dev_ctx_raw,
                hidden_dim as i32,
                &beta,
                self.dev_act_out,
                hidden_dim as i32,
            );
            if status_out != 0 { bail!("cublasSgemm_v2 Out failed with status {}", status_out); }

            // Copy final attention output to host
            cudaMemcpyAsync(
                attn_out.as_mut_ptr() as *mut c_void,
                self.dev_act_out as *const c_void,
                q_bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );

            let mut q_rope = vec![0.0f32; seq_len * hidden_dim];
            let mut k_rope = vec![0.0f32; seq_len * kv_dim];
            let mut v_data = vec![0.0f32; seq_len * kv_dim];
            let mut attn_probs = vec![0.0f32; num_heads * seq_len * seq_len];
            let mut attn_concat = vec![0.0f32; seq_len * hidden_dim];

            cudaMemcpyAsync(q_rope.as_mut_ptr() as *mut c_void, self.dev_q_raw as *const c_void, q_bytes, CUDA_MEMCPY_DEVICE_TO_HOST, std::ptr::null_mut());
            cudaMemcpyAsync(k_rope.as_mut_ptr() as *mut c_void, self.dev_k_raw as *const c_void, kv_bytes, CUDA_MEMCPY_DEVICE_TO_HOST, std::ptr::null_mut());
            cudaMemcpyAsync(v_data.as_mut_ptr() as *mut c_void, self.dev_v_raw as *const c_void, kv_bytes, CUDA_MEMCPY_DEVICE_TO_HOST, std::ptr::null_mut());
            cudaMemcpyAsync(attn_probs.as_mut_ptr() as *mut c_void, self.dev_scores as *const c_void, scores_bytes, CUDA_MEMCPY_DEVICE_TO_HOST, std::ptr::null_mut());
            cudaMemcpyAsync(attn_concat.as_mut_ptr() as *mut c_void, self.dev_ctx_raw as *const c_void, q_bytes, CUDA_MEMCPY_DEVICE_TO_HOST, std::ptr::null_mut());

            cudaDeviceSynchronize();

            Ok((q_rope, k_rope, v_data, attn_probs, attn_concat))
        }
    }

    pub fn gpu_batched_attention_backward(
        &mut self,
        d_x_res: &[f32],
        normed1: &[f32],
        q_rope: &[f32],
        k_rope: &[f32],
        v: &[f32],
        attn_probs: &[f32],
        attn_concat: &[f32],
        q_id: usize,
        k_id: usize,
        v_id: usize,
        out_id: usize,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        d_normed1: &mut [f32],
    ) -> Result<()> {
        let hidden_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q_bytes = seq_len * hidden_dim * std::mem::size_of::<f32>();
        let kv_bytes = seq_len * kv_dim * std::mem::size_of::<f32>();
        let scores_bytes = num_heads * seq_len * seq_len * std::mem::size_of::<f32>();

        unsafe {
            // 1. Out Projection Backward
            cudaMemcpyAsync(self.dev_act_aux as *mut c_void, d_x_res.as_ptr() as *const c_void, q_bytes, CUDA_MEMCPY_HOST_TO_DEVICE, std::ptr::null_mut());
            cudaMemcpyAsync(self.dev_ctx_raw as *mut c_void, attn_concat.as_ptr() as *const c_void, q_bytes, CUDA_MEMCPY_HOST_TO_DEVICE, std::ptr::null_mut());

            let meta_out = &self.params_meta[out_id];
            let alpha = 1.0f32;
            let beta_one = 1.0f32;
            let beta_zero = 0.0f32;

            // dWo += attn_concat^T * d_x_res directly into dev_grads:
            let status_dwo = cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                hidden_dim as i32,
                hidden_dim as i32,
                seq_len as i32,
                &alpha,
                self.dev_act_aux,
                hidden_dim as i32,
                self.dev_ctx_raw,
                hidden_dim as i32,
                &beta_one,
                self.dev_grads.add(meta_out.offset),
                hidden_dim as i32,
            );
            if status_dwo != 0 { bail!("cublasSgemm_v2 dWo failed: status {}", status_dwo); }

            // Compute d_Ctx = d_x_res * Wo^T -> dev_d_ctx_raw
            let status_dctx = cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                hidden_dim as i32,
                seq_len as i32,
                hidden_dim as i32,
                &alpha,
                self.dev_weights.add(meta_out.offset),
                hidden_dim as i32,
                self.dev_act_aux,
                hidden_dim as i32,
                &beta_zero,
                self.dev_d_ctx_raw,
                hidden_dim as i32,
            );
            if status_dctx != 0 { bail!("cublasSgemm_v2 dCtx failed: status {}", status_dctx); }

            // Permute d_Ctx -> dev_d_ctx_perm [H, T, D]
            launch_transpose_0213(self.dev_d_ctx_raw, self.dev_d_ctx_perm, seq_len, num_heads, head_dim, std::ptr::null_mut());

            // Upload attn_probs to dev_scores, v to dev_v_raw
            cudaMemcpyAsync(self.dev_scores as *mut c_void, attn_probs.as_ptr() as *const c_void, scores_bytes, CUDA_MEMCPY_HOST_TO_DEVICE, std::ptr::null_mut());
            cudaMemcpyAsync(self.dev_v_raw as *mut c_void, v.as_ptr() as *const c_void, kv_bytes, CUDA_MEMCPY_HOST_TO_DEVICE, std::ptr::null_mut());
            launch_repeat_kv_heads(self.dev_v_raw, self.dev_v_perm, seq_len, num_heads, num_kv_heads, head_dim, std::ptr::null_mut());

            // 2. Batched GEMMs:
            // d_V_perm = d_Ctx_perm * Probs^T -> dev_d_v_perm [H, T, D]
            let status_dv = cublasSgemmStridedBatched(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                head_dim as i32,
                seq_len as i32,
                seq_len as i32,
                &alpha,
                self.dev_d_ctx_perm,
                head_dim as i32,
                (seq_len * head_dim) as i64,
                self.dev_scores,
                seq_len as i32,
                (seq_len * seq_len) as i64,
                &beta_zero,
                self.dev_d_v_perm,
                head_dim as i32,
                (seq_len * head_dim) as i64,
                num_heads as i32,
            );
            if status_dv != 0 { bail!("cublasSgemmStridedBatched dV failed: status {}", status_dv); }

            // d_Scores_raw = V_perm^T * d_Ctx_perm -> dev_d_scores_raw [H, T, T]
            let status_ds_raw = cublasSgemmStridedBatched(
                self.cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                seq_len as i32,
                seq_len as i32,
                head_dim as i32,
                &alpha,
                self.dev_v_perm,
                head_dim as i32,
                (seq_len * head_dim) as i64,
                self.dev_d_ctx_perm,
                head_dim as i32,
                (seq_len * head_dim) as i64,
                &beta_zero,
                self.dev_d_scores_raw,
                seq_len as i32,
                (seq_len * seq_len) as i64,
                num_heads as i32,
            );
            if status_ds_raw != 0 { bail!("cublasSgemmStridedBatched dScores_raw failed: status {}", status_ds_raw); }

            // 3. Causal Softmax Backward
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            launch_causal_softmax_backward(
                self.dev_d_scores_raw,
                self.dev_scores,
                self.dev_d_scores,
                scale,
                num_heads,
                seq_len,
                std::ptr::null_mut(),
            );

            // 4. Batched GEMMs for dQ and dK:
            cudaMemcpyAsync(self.dev_q_raw as *mut c_void, q_rope.as_ptr() as *const c_void, q_bytes, CUDA_MEMCPY_HOST_TO_DEVICE, std::ptr::null_mut());
            cudaMemcpyAsync(self.dev_k_raw as *mut c_void, k_rope.as_ptr() as *const c_void, kv_bytes, CUDA_MEMCPY_HOST_TO_DEVICE, std::ptr::null_mut());

            launch_transpose_0213(self.dev_q_raw, self.dev_q_perm, seq_len, num_heads, head_dim, std::ptr::null_mut());
            launch_repeat_kv_heads(self.dev_k_raw, self.dev_k_perm, seq_len, num_heads, num_kv_heads, head_dim, std::ptr::null_mut());

            // d_Q_perm = K_perm * d_Scores -> dev_d_q_perm [H, T, D]
            let status_dq = cublasSgemmStridedBatched(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                head_dim as i32,
                seq_len as i32,
                seq_len as i32,
                &alpha,
                self.dev_k_perm,
                head_dim as i32,
                (seq_len * head_dim) as i64,
                self.dev_d_scores,
                seq_len as i32,
                (seq_len * seq_len) as i64,
                &beta_zero,
                self.dev_d_q_perm,
                head_dim as i32,
                (seq_len * head_dim) as i64,
                num_heads as i32,
            );
            if status_dq != 0 { bail!("cublasSgemmStridedBatched dQ failed: status {}", status_dq); }

            // d_K_perm = Q_perm * d_Scores^T -> dev_d_k_perm [H, T, D]
            let status_dk = cublasSgemmStridedBatched(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                head_dim as i32,
                seq_len as i32,
                seq_len as i32,
                &alpha,
                self.dev_q_perm,
                head_dim as i32,
                (seq_len * head_dim) as i64,
                self.dev_d_scores,
                seq_len as i32,
                (seq_len * seq_len) as i64,
                &beta_zero,
                self.dev_d_k_perm,
                head_dim as i32,
                (seq_len * head_dim) as i64,
                num_heads as i32,
            );
            if status_dk != 0 { bail!("cublasSgemmStridedBatched dK failed: status {}", status_dk); }

            // 5. Permutation & KV Reduction:
            launch_transpose_102_to_012(self.dev_d_q_perm, self.dev_d_q_raw, num_heads, seq_len, head_dim, std::ptr::null_mut());
            launch_reduce_kv_heads_grad(self.dev_d_k_perm, self.dev_d_k_raw, seq_len, num_heads, num_kv_heads, head_dim, std::ptr::null_mut());
            launch_reduce_kv_heads_grad(self.dev_d_v_perm, self.dev_d_v_raw, seq_len, num_heads, num_kv_heads, head_dim, std::ptr::null_mut());

            // 6. RoPE Backward
            launch_rope(self.dev_d_q_raw, self.dev_cos_cache, self.dev_sin_cache, seq_len, num_heads, head_dim, -1, std::ptr::null_mut());
            launch_rope(self.dev_d_k_raw, self.dev_cos_cache, self.dev_sin_cache, seq_len, num_kv_heads, head_dim, -1, std::ptr::null_mut());

            // 7. Backward Weight GEMMs into dev_grads:
            let normed_bytes = normed1.len() * std::mem::size_of::<f32>();
            cudaMemcpyAsync(self.dev_act_in as *mut c_void, normed1.as_ptr() as *const c_void, normed_bytes, CUDA_MEMCPY_HOST_TO_DEVICE, std::ptr::null_mut());

            let meta_q = &self.params_meta[q_id];
            let meta_k = &self.params_meta[k_id];
            let meta_v = &self.params_meta[v_id];

            // dWq += normed1^T * d_Q
            cublasSgemm_v2(self.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, hidden_dim as i32, hidden_dim as i32, seq_len as i32, &alpha, self.dev_d_q_raw, hidden_dim as i32, self.dev_act_in, hidden_dim as i32, &beta_one, self.dev_grads.add(meta_q.offset), hidden_dim as i32);
            // dWk += normed1^T * d_K
            cublasSgemm_v2(self.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, kv_dim as i32, hidden_dim as i32, seq_len as i32, &alpha, self.dev_d_k_raw, kv_dim as i32, self.dev_act_in, hidden_dim as i32, &beta_one, self.dev_grads.add(meta_k.offset), kv_dim as i32);
            // dWv += normed1^T * d_V
            cublasSgemm_v2(self.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, kv_dim as i32, hidden_dim as i32, seq_len as i32, &alpha, self.dev_d_v_raw, kv_dim as i32, self.dev_act_in, hidden_dim as i32, &beta_one, self.dev_grads.add(meta_v.offset), kv_dim as i32);

            // 8. Backward Data GEMMs to compute d_normed1:
            // d_normed1 = d_Q * Wq^T + d_K * Wk^T + d_V * Wv^T
            cublasSgemm_v2(self.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden_dim as i32, seq_len as i32, hidden_dim as i32, &alpha, self.dev_weights.add(meta_q.offset), hidden_dim as i32, self.dev_d_q_raw, hidden_dim as i32, &beta_zero, self.dev_act_out, hidden_dim as i32);
            cublasSgemm_v2(self.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden_dim as i32, seq_len as i32, kv_dim as i32, &alpha, self.dev_weights.add(meta_k.offset), kv_dim as i32, self.dev_d_k_raw, kv_dim as i32, &beta_zero, self.dev_act_aux, hidden_dim as i32);
            launch_vector_add(self.dev_act_aux, self.dev_act_out, seq_len * hidden_dim, std::ptr::null_mut());

            cublasSgemm_v2(self.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden_dim as i32, seq_len as i32, kv_dim as i32, &alpha, self.dev_weights.add(meta_v.offset), kv_dim as i32, self.dev_d_v_raw, kv_dim as i32, &beta_zero, self.dev_act_aux, hidden_dim as i32);
            launch_vector_add(self.dev_act_aux, self.dev_act_out, seq_len * hidden_dim, std::ptr::null_mut());

            cudaMemcpyAsync(d_normed1.as_mut_ptr() as *mut c_void, self.dev_act_out as *const c_void, q_bytes, CUDA_MEMCPY_DEVICE_TO_HOST, std::ptr::null_mut());
            cudaDeviceSynchronize();

            Ok(())
        }
    }

    pub fn gpu_fused_cross_entropy(
        &mut self,
        normed_final: &[f32],
        targets: &[u16],
        mask: &[f32],
        wte_param_id: usize,
        seq_len: usize,
        hidden_dim: usize,
        vocab_size: usize,
        d_normed_final: &mut [f32],
        grad_scale: f32,
    ) -> Result<f32> {
        let normed_bytes = seq_len * hidden_dim * std::mem::size_of::<f32>();
        let targets_bytes = seq_len * std::mem::size_of::<u16>();
        let mask_bytes = seq_len * std::mem::size_of::<f32>();

        if seq_len > self.target_mask_capacity {
            bail!("seq_len {} exceeds target_mask_capacity {}", seq_len, self.target_mask_capacity);
        }

        let mut active_count = 0.0f32;
        for &m in mask {
            active_count += m;
        }

        unsafe {
            cudaMemcpyAsync(self.dev_act_in as *mut c_void, normed_final.as_ptr() as *const c_void, normed_bytes, CUDA_MEMCPY_HOST_TO_DEVICE, std::ptr::null_mut());
            cudaMemcpyAsync(self.dev_targets as *mut c_void, targets.as_ptr() as *const c_void, targets_bytes, CUDA_MEMCPY_HOST_TO_DEVICE, std::ptr::null_mut());
            cudaMemcpyAsync(self.dev_mask as *mut c_void, mask.as_ptr() as *const c_void, mask_bytes, CUDA_MEMCPY_HOST_TO_DEVICE, std::ptr::null_mut());
            cudaMemsetAsync(self.dev_loss as *mut c_void, 0, std::mem::size_of::<f32>(), std::ptr::null_mut());

            let meta_wte = &self.params_meta[wte_param_id];
            let alpha = 1.0f32;
            let beta_zero = 0.0f32;

            // 1. Logits = normed_final * WTE^T
            let status_logits = cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                vocab_size as i32,
                seq_len as i32,
                hidden_dim as i32,
                &alpha,
                self.dev_weights.add(meta_wte.offset),
                hidden_dim as i32,
                self.dev_act_in,
                hidden_dim as i32,
                &beta_zero,
                self.dev_act_out,
                vocab_size as i32,
            );
            if status_logits != 0 { bail!("cublasSgemm_v2 Logits failed: status {}", status_logits); }

            // 2. Fused Cross-Entropy in VRAM
            launch_fused_cross_entropy(
                self.dev_act_out,
                self.dev_targets,
                self.dev_mask,
                self.dev_act_aux,
                self.dev_loss,
                seq_len,
                vocab_size,
                active_count,
                grad_scale,
                std::ptr::null_mut(),
            );

            // 3. dW_wte += d_logits^T * normed_final
            let beta_one = 1.0f32;
            let status_d_wte = cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                hidden_dim as i32,
                vocab_size as i32,
                seq_len as i32,
                &alpha,
                self.dev_act_in,
                hidden_dim as i32,
                self.dev_act_aux,
                vocab_size as i32,
                &beta_one,
                self.dev_grads.add(meta_wte.offset),
                hidden_dim as i32,
            );
            if status_d_wte != 0 { bail!("cublasSgemm_v2 dW_wte failed: status {}", status_d_wte); }

            // 4. d_normed_final = d_logits * WTE
            let status_d_normed = cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                hidden_dim as i32,
                seq_len as i32,
                vocab_size as i32,
                &alpha,
                self.dev_weights.add(meta_wte.offset),
                hidden_dim as i32,
                self.dev_act_aux,
                vocab_size as i32,
                &beta_zero,
                self.dev_act_in,
                hidden_dim as i32,
            );
            if status_d_normed != 0 { bail!("cublasSgemm_v2 d_normed_final failed: status {}", status_d_normed); }

            cudaMemcpyAsync(
                d_normed_final.as_mut_ptr() as *mut c_void,
                self.dev_act_in as *const c_void,
                normed_bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );

            let mut loss_host = 0.0f32;
            cudaMemcpyAsync(
                &mut loss_host as *mut f32 as *mut c_void,
                self.dev_loss as *const c_void,
                std::mem::size_of::<f32>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );

            cudaDeviceSynchronize();

            Ok(loss_host)
        }
    }

    /// Download dev_act_out buffer (e.g. logits from forward / cross-entropy) to host
    pub fn download_act_out(&self, host_out: &mut [f32]) -> Result<()> {
        let bytes = host_out.len() * std::mem::size_of::<f32>();
        unsafe {
            cudaMemcpyAsync(
                host_out.as_mut_ptr() as *mut c_void,
                self.dev_act_out as *const c_void,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );
            cudaDeviceSynchronize();
        }
        Ok(())
    }

    /// GPU RMSNorm Forward: 100% in VRAM via rmsnorm_forward_kernel
    pub fn gpu_rmsnorm_forward(
        &mut self,
        x: &[f32],
        param_id: usize,
        out: &mut [f32],
        rstd: &mut [f32],
        seq_len: usize,
        hidden_dim: usize,
        eps: f32,
    ) -> Result<()> {
        let meta = &self.params_meta[param_id];
        if meta.len != hidden_dim {
            bail!("RMSNorm param {} len mismatch: {} != {}", param_id, meta.len, hidden_dim);
        }
        let total = seq_len * hidden_dim;
        let x_bytes = total * std::mem::size_of::<f32>();
        if total > self.act_in_capacity || total > self.act_out_capacity || seq_len > 16384 {
            bail!("RMSNorm dimensions exceed workspace capacity");
        }

        unsafe {
            cudaMemcpyAsync(
                self.dev_act_in as *mut c_void,
                x.as_ptr() as *const c_void,
                x_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );

            let weight_ptr = self.dev_weights.add(meta.offset);
            launch_rmsnorm_forward(
                self.dev_act_in,
                weight_ptr,
                self.dev_act_out,
                self.dev_rstd,
                seq_len,
                hidden_dim,
                eps,
                std::ptr::null_mut(),
            );

            cudaMemcpyAsync(
                out.as_mut_ptr() as *mut c_void,
                self.dev_act_out as *const c_void,
                x_bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );
            cudaMemcpyAsync(
                rstd.as_mut_ptr() as *mut c_void,
                self.dev_rstd as *const c_void,
                seq_len * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );
            cudaDeviceSynchronize();
        }
        Ok(())
    }

    /// GPU RMSNorm Backward: 100% in VRAM, directly accumulates dweight into dev_grads
    pub fn gpu_rmsnorm_backward(
        &mut self,
        dout: &[f32],
        x: &[f32],
        rstd: &[f32],
        param_id: usize,
        dx: &mut [f32],
        seq_len: usize,
        hidden_dim: usize,
    ) -> Result<()> {
        let meta = &self.params_meta[param_id];
        let total = seq_len * hidden_dim;
        let bytes = total * std::mem::size_of::<f32>();
        if total > self.act_in_capacity || total > self.act_out_capacity || seq_len > 16384 {
            bail!("RMSNorm backward dimensions exceed workspace");
        }

        unsafe {
            cudaMemcpyAsync(
                self.dev_act_out as *mut c_void,
                dout.as_ptr() as *const c_void,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );
            cudaMemcpyAsync(
                self.dev_act_in as *mut c_void,
                x.as_ptr() as *const c_void,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );
            cudaMemcpyAsync(
                self.dev_rstd as *mut c_void,
                rstd.as_ptr() as *const c_void,
                seq_len * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );

            let weight_ptr = self.dev_weights.add(meta.offset);
            let dweight_accum_ptr = self.dev_grads.add(meta.offset);

            launch_rmsnorm_backward(
                self.dev_act_out,
                self.dev_act_in,
                weight_ptr,
                self.dev_rstd,
                self.dev_act_aux,
                dweight_accum_ptr,
                seq_len,
                hidden_dim,
                std::ptr::null_mut(),
            );

            cudaMemcpyAsync(
                dx.as_mut_ptr() as *mut c_void,
                self.dev_act_aux as *const c_void,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );
            cudaDeviceSynchronize();
        }
        Ok(())
    }

    /// Shared Expert Forward directly in VRAM (initializes dev_moe_out)
    pub fn gpu_moe_forward_shared(
        &mut self,
        normed2: &[f32],
        gate_id: usize,
        up_id: usize,
        down_id: usize,
        seq_len: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let gate_meta = self.params_meta[gate_id];
        let up_meta = self.params_meta[up_id];
        let down_meta = self.params_meta[down_id];

        let n_in = seq_len * hidden_dim;
        let n_mid = seq_len * intermediate_dim;

        if n_in > self.act_in_capacity || n_mid > self.moe_dim_capacity {
            bail!("MoE shared dimensions exceed workspace");
        }

        let mut host_gate = vec![0.0f32; n_mid];
        let mut host_up = vec![0.0f32; n_mid];
        let mut host_act = vec![0.0f32; n_mid];
        let mut host_out = vec![0.0f32; n_in];

        unsafe {
            // Upload normed2 to dev_act_in
            cudaMemcpyAsync(
                self.dev_act_in as *mut c_void,
                normed2.as_ptr() as *const c_void,
                n_in * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );

            let alpha = 1.0f32;
            let beta_zero = 0.0f32;

            // 1. Gate GEMM: [seq_len, hidden_dim] * [hidden_dim, intermediate_dim] -> [seq_len, intermediate_dim]
            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                intermediate_dim as i32,
                seq_len as i32,
                hidden_dim as i32,
                &alpha,
                self.dev_weights.add(gate_meta.offset),
                intermediate_dim as i32,
                self.dev_act_in,
                hidden_dim as i32,
                &beta_zero,
                self.dev_shared_gate,
                intermediate_dim as i32,
            );

            // 2. Up GEMM: [seq_len, hidden_dim] * [hidden_dim, intermediate_dim] -> [seq_len, intermediate_dim]
            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                intermediate_dim as i32,
                seq_len as i32,
                hidden_dim as i32,
                &alpha,
                self.dev_weights.add(up_meta.offset),
                intermediate_dim as i32,
                self.dev_act_in,
                hidden_dim as i32,
                &beta_zero,
                self.dev_shared_up,
                intermediate_dim as i32,
            );

            // 3. SwiGLU in VRAM
            launch_swiglu_forward(
                self.dev_shared_gate,
                self.dev_shared_up,
                self.dev_shared_act,
                n_mid,
                std::ptr::null_mut(),
            );

            // 4. Down GEMM into dev_moe_out: [seq_len, intermediate_dim] * [intermediate_dim, hidden_dim] -> [seq_len, hidden_dim]
            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                hidden_dim as i32,
                seq_len as i32,
                intermediate_dim as i32,
                &alpha,
                self.dev_weights.add(down_meta.offset),
                hidden_dim as i32,
                self.dev_shared_act,
                intermediate_dim as i32,
                &beta_zero,
                self.dev_moe_out,
                hidden_dim as i32,
            );

            // Download activations for layer backward stash
            cudaMemcpyAsync(
                host_gate.as_mut_ptr() as *mut c_void,
                self.dev_shared_gate as *const c_void,
                n_mid * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );
            cudaMemcpyAsync(
                host_up.as_mut_ptr() as *mut c_void,
                self.dev_shared_up as *const c_void,
                n_mid * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );
            cudaMemcpyAsync(
                host_act.as_mut_ptr() as *mut c_void,
                self.dev_shared_act as *const c_void,
                n_mid * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );
            cudaMemcpyAsync(
                host_out.as_mut_ptr() as *mut c_void,
                self.dev_moe_out as *const c_void,
                n_in * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );
        }

        Ok((host_gate, host_up, host_act, host_out))
    }

    /// Routed Expert Forward directly in VRAM (accumulates via atomic scatter-add into dev_moe_out)
    pub fn gpu_moe_forward_expert(
        &mut self,
        token_indices: &[i32],
        weights: &[f32],
        gate_id: usize,
        up_id: usize,
        down_id: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let n_tokens = token_indices.len();
        if n_tokens == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        }

        let gate_meta = self.params_meta[gate_id];
        let up_meta = self.params_meta[up_id];
        let down_meta = self.params_meta[down_id];

        let n_in = n_tokens * hidden_dim;
        let n_mid = n_tokens * intermediate_dim;

        let mut host_x = vec![0.0f32; n_in];
        let mut host_gate = vec![0.0f32; n_mid];
        let mut host_up = vec![0.0f32; n_mid];
        let mut host_act = vec![0.0f32; n_mid];
        let mut host_out = vec![0.0f32; n_in];

        unsafe {
            cudaMemcpyAsync(
                self.dev_exp_indices as *mut c_void,
                token_indices.as_ptr() as *const c_void,
                n_tokens * std::mem::size_of::<i32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );
            cudaMemcpyAsync(
                self.dev_exp_weights as *mut c_void,
                weights.as_ptr() as *const c_void,
                n_tokens * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );

            // 1. Gather tokens from dev_act_in (holding normed2) into dev_exp_x in VRAM
            launch_moe_gather(
                self.dev_act_in,
                self.dev_exp_indices,
                self.dev_exp_x,
                n_tokens,
                hidden_dim,
                std::ptr::null_mut(),
            );

            let alpha = 1.0f32;
            let beta_zero = 0.0f32;

            // 2. Gate & Up GEMMs
            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                intermediate_dim as i32,
                n_tokens as i32,
                hidden_dim as i32,
                &alpha,
                self.dev_weights.add(gate_meta.offset),
                intermediate_dim as i32,
                self.dev_exp_x,
                hidden_dim as i32,
                &beta_zero,
                self.dev_exp_gate,
                intermediate_dim as i32,
            );

            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                intermediate_dim as i32,
                n_tokens as i32,
                hidden_dim as i32,
                &alpha,
                self.dev_weights.add(up_meta.offset),
                intermediate_dim as i32,
                self.dev_exp_x,
                hidden_dim as i32,
                &beta_zero,
                self.dev_exp_up,
                intermediate_dim as i32,
            );

            // 3. SwiGLU in VRAM
            launch_swiglu_forward(
                self.dev_exp_gate,
                self.dev_exp_up,
                self.dev_exp_act,
                n_mid,
                std::ptr::null_mut(),
            );

            // 4. Down GEMM into dev_exp_out
            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                hidden_dim as i32,
                n_tokens as i32,
                intermediate_dim as i32,
                &alpha,
                self.dev_weights.add(down_meta.offset),
                hidden_dim as i32,
                self.dev_exp_act,
                intermediate_dim as i32,
                &beta_zero,
                self.dev_exp_out,
                hidden_dim as i32,
            );

            // 5. Scatter-Add into dev_moe_out in VRAM
            launch_moe_scatter_add(
                self.dev_exp_out,
                self.dev_exp_indices,
                self.dev_exp_weights,
                self.dev_moe_out,
                n_tokens,
                hidden_dim,
                std::ptr::null_mut(),
            );

            // Download activations for backward stash
            cudaMemcpyAsync(
                host_x.as_mut_ptr() as *mut c_void,
                self.dev_exp_x as *const c_void,
                n_in * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );
            cudaMemcpyAsync(
                host_gate.as_mut_ptr() as *mut c_void,
                self.dev_exp_gate as *const c_void,
                n_mid * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );
            cudaMemcpyAsync(
                host_up.as_mut_ptr() as *mut c_void,
                self.dev_exp_up as *const c_void,
                n_mid * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );
            cudaMemcpyAsync(
                host_act.as_mut_ptr() as *mut c_void,
                self.dev_exp_act as *const c_void,
                n_mid * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );
            cudaMemcpyAsync(
                host_out.as_mut_ptr() as *mut c_void,
                self.dev_exp_out as *const c_void,
                n_in * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );
        }

        Ok((host_x, host_gate, host_up, host_act, host_out))
    }

    /// Download total MoE output from dev_moe_out to host
    pub fn gpu_moe_finalize_forward(&self, total_out: &mut [f32]) -> Result<()> {
        let bytes = total_out.len() * std::mem::size_of::<f32>();
        unsafe {
            cudaMemcpyAsync(
                total_out.as_mut_ptr() as *mut c_void,
                self.dev_moe_out as *const c_void,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );
            cudaDeviceSynchronize();
        }
        Ok(())
    }

    /// Initialize MoE backward by uploading d_moe to dev_act_aux and clearing dev_act_out (dev_d_normed2)
    pub fn gpu_moe_init_backward(&mut self, d_moe: &[f32]) -> Result<()> {
        let bytes = d_moe.len() * std::mem::size_of::<f32>();
        unsafe {
            cudaMemcpyAsync(
                self.dev_act_aux as *mut c_void,
                d_moe.as_ptr() as *const c_void,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );
            // Clear dev_act_out which accumulates d_normed2
            cudaMemsetAsync(self.dev_act_out as *mut c_void, 0, bytes, std::ptr::null_mut());
        }
        Ok(())
    }

    /// Routed Expert Backward directly in VRAM (accumulates into dev_grads and dev_act_out)
    pub fn gpu_moe_backward_expert(
        &mut self,
        token_indices: &[i32],
        weights: &[f32],
        exp_x: &[f32],
        exp_gate: &[f32],
        exp_up: &[f32],
        exp_act: &[f32],
        exp_out: &[f32],
        gate_id: usize,
        up_id: usize,
        down_id: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Result<Vec<f32>> {
        let n_tokens = token_indices.len();
        if n_tokens == 0 {
            return Ok(Vec::new());
        }

        let gate_meta = self.params_meta[gate_id];
        let up_meta = self.params_meta[up_id];
        let down_meta = self.params_meta[down_id];

        let n_in = n_tokens * hidden_dim;
        let n_mid = n_tokens * intermediate_dim;
        let mut host_scalars = vec![0.0f32; n_tokens];

        unsafe {
            cudaMemcpyAsync(
                self.dev_exp_indices as *mut c_void,
                token_indices.as_ptr() as *const c_void,
                n_tokens * std::mem::size_of::<i32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );
            cudaMemcpyAsync(
                self.dev_exp_weights as *mut c_void,
                weights.as_ptr() as *const c_void,
                n_tokens * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );
            cudaMemcpyAsync(
                self.dev_exp_out as *mut c_void,
                exp_out.as_ptr() as *const c_void,
                n_in * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );

            // 1. Gather weighted d_moe -> dev_exp_dy
            launch_moe_backward_gather_weighted(
                self.dev_act_aux, // holding d_moe
                self.dev_exp_indices,
                self.dev_exp_weights,
                self.dev_exp_dy,
                n_tokens,
                hidden_dim,
                std::ptr::null_mut(),
            );

            // 2. Parallel router dot product in VRAM
            launch_moe_router_dot(
                self.dev_exp_out,
                self.dev_act_aux,
                self.dev_exp_indices,
                self.dev_exp_scalars,
                n_tokens,
                hidden_dim,
                std::ptr::null_mut(),
            );

            // 3. Upload act
            cudaMemcpyAsync(
                self.dev_exp_act as *mut c_void,
                exp_act.as_ptr() as *const c_void,
                n_mid * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );

            let alpha = 1.0f32;
            let beta_one = 1.0f32;
            let beta_zero = 0.0f32;

            // 4. Down proj dW GEMM: dW_down += act^T * dY_e
            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                hidden_dim as i32,
                intermediate_dim as i32,
                n_tokens as i32,
                &alpha,
                self.dev_exp_dy,
                hidden_dim as i32,
                self.dev_exp_act,
                intermediate_dim as i32,
                &beta_one,
                self.dev_grads.add(down_meta.offset),
                hidden_dim as i32,
            );

            // 5. Down proj dAct GEMM: dAct_e = dY_e * W_down^T
            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                intermediate_dim as i32,
                n_tokens as i32,
                hidden_dim as i32,
                &alpha,
                self.dev_weights.add(down_meta.offset),
                hidden_dim as i32,
                self.dev_exp_dy,
                hidden_dim as i32,
                &beta_zero,
                self.dev_exp_dact,
                intermediate_dim as i32,
            );

            // 6. Upload gate & up
            cudaMemcpyAsync(
                self.dev_exp_gate as *mut c_void,
                exp_gate.as_ptr() as *const c_void,
                n_mid * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );
            cudaMemcpyAsync(
                self.dev_exp_up as *mut c_void,
                exp_up.as_ptr() as *const c_void,
                n_mid * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );

            // 7. SwiGLU backward in VRAM
            launch_swiglu_backward(
                self.dev_exp_dact,
                self.dev_exp_gate,
                self.dev_exp_up,
                self.dev_exp_dgate,
                self.dev_exp_dup,
                n_mid,
                std::ptr::null_mut(),
            );

            // 8. Upload x
            cudaMemcpyAsync(
                self.dev_exp_x as *mut c_void,
                exp_x.as_ptr() as *const c_void,
                n_in * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );

            // 9. Gate & Up proj dW GEMMs
            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                intermediate_dim as i32,
                hidden_dim as i32,
                n_tokens as i32,
                &alpha,
                self.dev_exp_dgate,
                intermediate_dim as i32,
                self.dev_exp_x,
                hidden_dim as i32,
                &beta_one,
                self.dev_grads.add(gate_meta.offset),
                intermediate_dim as i32,
            );

            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                intermediate_dim as i32,
                hidden_dim as i32,
                n_tokens as i32,
                &alpha,
                self.dev_exp_dup,
                intermediate_dim as i32,
                self.dev_exp_x,
                hidden_dim as i32,
                &beta_one,
                self.dev_grads.add(up_meta.offset),
                intermediate_dim as i32,
            );

            // 10. Gate & Up proj dX GEMMs
            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                hidden_dim as i32,
                n_tokens as i32,
                intermediate_dim as i32,
                &alpha,
                self.dev_weights.add(gate_meta.offset),
                intermediate_dim as i32,
                self.dev_exp_dgate,
                intermediate_dim as i32,
                &beta_zero,
                self.dev_exp_dx,
                hidden_dim as i32,
            );

            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                hidden_dim as i32,
                n_tokens as i32,
                intermediate_dim as i32,
                &alpha,
                self.dev_weights.add(up_meta.offset),
                intermediate_dim as i32,
                self.dev_exp_dup,
                intermediate_dim as i32,
                &beta_zero,
                self.dev_exp_dx_tmp,
                hidden_dim as i32,
            );

            // 11. Scatter-Add grad to dev_act_out (holding d_normed2)
            launch_moe_scatter_add_grad(
                self.dev_exp_dx,
                self.dev_exp_dx_tmp,
                self.dev_exp_indices,
                self.dev_exp_weights,
                self.dev_act_out,
                n_tokens,
                hidden_dim,
                std::ptr::null_mut(),
            );

            // 12. Download router scalars
            cudaMemcpyAsync(
                host_scalars.as_mut_ptr() as *mut c_void,
                self.dev_exp_scalars as *const c_void,
                n_tokens * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );
        }

        Ok(host_scalars)
    }

    /// Shared Expert Backward directly in VRAM (accumulates into dev_grads and dev_act_out)
    pub fn gpu_moe_backward_shared(
        &mut self,
        shared_act: &[f32],
        shared_gate: &[f32],
        shared_up: &[f32],
        normed2: &[f32],
        gate_id: usize,
        up_id: usize,
        down_id: usize,
        seq_len: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Result<()> {
        let gate_meta = self.params_meta[gate_id];
        let up_meta = self.params_meta[up_id];
        let down_meta = self.params_meta[down_id];

        let n_in = seq_len * hidden_dim;
        let n_mid = seq_len * intermediate_dim;

        unsafe {
            // Upload shared_act
            cudaMemcpyAsync(
                self.dev_shared_act as *mut c_void,
                shared_act.as_ptr() as *const c_void,
                n_mid * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );

            let alpha = 1.0f32;
            let beta_one = 1.0f32;
            let beta_zero = 0.0f32;

            // 1. Down proj dW GEMM: dW_down += shared_act^T * d_moe (dev_act_aux)
            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                hidden_dim as i32,
                intermediate_dim as i32,
                seq_len as i32,
                &alpha,
                self.dev_act_aux,
                hidden_dim as i32,
                self.dev_shared_act,
                intermediate_dim as i32,
                &beta_one,
                self.dev_grads.add(down_meta.offset),
                hidden_dim as i32,
            );

            // 2. Down proj dAct GEMM: d_shared_act = d_moe * W_down^T
            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                intermediate_dim as i32,
                seq_len as i32,
                hidden_dim as i32,
                &alpha,
                self.dev_weights.add(down_meta.offset),
                hidden_dim as i32,
                self.dev_act_aux,
                hidden_dim as i32,
                &beta_zero,
                self.dev_exp_dact,
                intermediate_dim as i32,
            );

            // 3. Upload shared_gate & shared_up
            cudaMemcpyAsync(
                self.dev_shared_gate as *mut c_void,
                shared_gate.as_ptr() as *const c_void,
                n_mid * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );
            cudaMemcpyAsync(
                self.dev_shared_up as *mut c_void,
                shared_up.as_ptr() as *const c_void,
                n_mid * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );

            // 4. SwiGLU backward in VRAM
            launch_swiglu_backward(
                self.dev_exp_dact,
                self.dev_shared_gate,
                self.dev_shared_up,
                self.dev_exp_dgate,
                self.dev_exp_dup,
                n_mid,
                std::ptr::null_mut(),
            );

            // 5. Upload normed2
            cudaMemcpyAsync(
                self.dev_act_in as *mut c_void,
                normed2.as_ptr() as *const c_void,
                n_in * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );

            // 6. Gate & Up proj dW GEMMs
            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                intermediate_dim as i32,
                hidden_dim as i32,
                seq_len as i32,
                &alpha,
                self.dev_exp_dgate,
                intermediate_dim as i32,
                self.dev_act_in,
                hidden_dim as i32,
                &beta_one,
                self.dev_grads.add(gate_meta.offset),
                intermediate_dim as i32,
            );

            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                intermediate_dim as i32,
                hidden_dim as i32,
                seq_len as i32,
                &alpha,
                self.dev_exp_dup,
                intermediate_dim as i32,
                self.dev_act_in,
                hidden_dim as i32,
                &beta_one,
                self.dev_grads.add(up_meta.offset),
                intermediate_dim as i32,
            );

            // 7. Gate & Up proj dX GEMMs directly into dev_act_out (accumulating d_normed2!)
            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                hidden_dim as i32,
                seq_len as i32,
                intermediate_dim as i32,
                &alpha,
                self.dev_weights.add(gate_meta.offset),
                intermediate_dim as i32,
                self.dev_exp_dgate,
                intermediate_dim as i32,
                &beta_one,
                self.dev_act_out,
                hidden_dim as i32,
            );

            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                hidden_dim as i32,
                seq_len as i32,
                intermediate_dim as i32,
                &alpha,
                self.dev_weights.add(up_meta.offset),
                intermediate_dim as i32,
                self.dev_exp_dup,
                intermediate_dim as i32,
                &beta_one,
                self.dev_act_out,
                hidden_dim as i32,
            );
        }

        Ok(())
    }

    /// Explicitly synchronize the CUDA stream with host CPU
    pub fn sync_stream(&self) -> Result<()> {
        unsafe {
            let status = cudaDeviceSynchronize();
            if status != 0 {
                bail!("cudaDeviceSynchronize failed with status {}", status);
            }
        }
        Ok(())
    }

    /// Router Backward directly in VRAM and finalize d_normed2
    pub fn gpu_moe_backward_router(
        &mut self,
        normed2: &[f32],
        d_router_logits: &[f32],
        router_id: usize,
        seq_len: usize,
        hidden_dim: usize,
        num_experts: usize,
        d_normed2_out: &mut [f32],
    ) -> Result<()> {
        let router_meta = self.params_meta[router_id];
        let logits_bytes = seq_len * num_experts * std::mem::size_of::<f32>();
        let normed_bytes = seq_len * hidden_dim * std::mem::size_of::<f32>();

        unsafe {
            // Upload d_router_logits into dev_shared_out
            cudaMemcpyAsync(
                self.dev_shared_out as *mut c_void,
                d_router_logits.as_ptr() as *const c_void,
                logits_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );
            // Upload normed2 into dev_act_in
            cudaMemcpyAsync(
                self.dev_act_in as *mut c_void,
                normed2.as_ptr() as *const c_void,
                normed_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                std::ptr::null_mut(),
            );

            let alpha = 1.0f32;
            let beta_one = 1.0f32;

            // 1. Router dW GEMM: dW_router += normed2^T * d_router_logits
            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                num_experts as i32,
                hidden_dim as i32,
                seq_len as i32,
                &alpha,
                self.dev_shared_out,
                num_experts as i32,
                self.dev_act_in,
                hidden_dim as i32,
                &beta_one,
                self.dev_grads.add(router_meta.offset),
                num_experts as i32,
            );

            // 2. Router dX GEMM directly into dev_act_out (accumulating d_normed2!)
            cublasSgemm_v2(
                self.cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                hidden_dim as i32,
                seq_len as i32,
                num_experts as i32,
                &alpha,
                self.dev_weights.add(router_meta.offset),
                num_experts as i32,
                self.dev_shared_out,
                num_experts as i32,
                &beta_one,
                self.dev_act_out,
                hidden_dim as i32,
            );

            // 3. Download final accumulated d_normed2 to host!
            cudaMemcpyAsync(
                d_normed2_out.as_mut_ptr() as *mut c_void,
                self.dev_act_out as *const c_void,
                normed_bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                std::ptr::null_mut(),
            );

            cudaDeviceSynchronize();
        }

        Ok(())
    }
}

impl Drop for GpuModelEngine {
    fn drop(&mut self) {
        unsafe {
            if !self.dev_weights.is_null() { cudaFree(self.dev_weights as *mut c_void); }
            if !self.dev_grads.is_null() { cudaFree(self.dev_grads as *mut c_void); }
            if !self.dev_adamw_m.is_null() { cudaFree(self.dev_adamw_m as *mut c_void); }
            if !self.dev_adamw_v.is_null() { cudaFree(self.dev_adamw_v as *mut c_void); }
            if !self.dev_act_in.is_null() { cudaFree(self.dev_act_in as *mut c_void); }
            if !self.dev_act_out.is_null() { cudaFree(self.dev_act_out as *mut c_void); }
            if !self.dev_act_aux.is_null() { cudaFree(self.dev_act_aux as *mut c_void); }

            if !self.dev_q_raw.is_null() { cudaFree(self.dev_q_raw as *mut c_void); }
            if !self.dev_k_raw.is_null() { cudaFree(self.dev_k_raw as *mut c_void); }
            if !self.dev_v_raw.is_null() { cudaFree(self.dev_v_raw as *mut c_void); }
            if !self.dev_q_perm.is_null() { cudaFree(self.dev_q_perm as *mut c_void); }
            if !self.dev_k_perm.is_null() { cudaFree(self.dev_k_perm as *mut c_void); }
            if !self.dev_v_perm.is_null() { cudaFree(self.dev_v_perm as *mut c_void); }
            if !self.dev_scores.is_null() { cudaFree(self.dev_scores as *mut c_void); }
            if !self.dev_ctx_perm.is_null() { cudaFree(self.dev_ctx_perm as *mut c_void); }
            if !self.dev_ctx_raw.is_null() { cudaFree(self.dev_ctx_raw as *mut c_void); }

            if !self.dev_d_ctx_raw.is_null() { cudaFree(self.dev_d_ctx_raw as *mut c_void); }
            if !self.dev_d_ctx_perm.is_null() { cudaFree(self.dev_d_ctx_perm as *mut c_void); }
            if !self.dev_d_scores.is_null() { cudaFree(self.dev_d_scores as *mut c_void); }
            if !self.dev_d_scores_raw.is_null() { cudaFree(self.dev_d_scores_raw as *mut c_void); }
            if !self.dev_d_q_perm.is_null() { cudaFree(self.dev_d_q_perm as *mut c_void); }
            if !self.dev_d_k_perm.is_null() { cudaFree(self.dev_d_k_perm as *mut c_void); }
            if !self.dev_d_v_perm.is_null() { cudaFree(self.dev_d_v_perm as *mut c_void); }
            if !self.dev_d_q_raw.is_null() { cudaFree(self.dev_d_q_raw as *mut c_void); }
            if !self.dev_d_k_raw.is_null() { cudaFree(self.dev_d_k_raw as *mut c_void); }
            if !self.dev_d_v_raw.is_null() { cudaFree(self.dev_d_v_raw as *mut c_void); }

            if !self.dev_cos_cache.is_null() { cudaFree(self.dev_cos_cache as *mut c_void); }
            if !self.dev_sin_cache.is_null() { cudaFree(self.dev_sin_cache as *mut c_void); }
            if !self.dev_loss.is_null() { cudaFree(self.dev_loss as *mut c_void); }
            if !self.dev_targets.is_null() { cudaFree(self.dev_targets as *mut c_void); }
            if !self.dev_mask.is_null() { cudaFree(self.dev_mask as *mut c_void); }

            if !self.dev_moe_out.is_null() { cudaFree(self.dev_moe_out as *mut c_void); }
            if !self.dev_shared_gate.is_null() { cudaFree(self.dev_shared_gate as *mut c_void); }
            if !self.dev_shared_up.is_null() { cudaFree(self.dev_shared_up as *mut c_void); }
            if !self.dev_shared_act.is_null() { cudaFree(self.dev_shared_act as *mut c_void); }
            if !self.dev_shared_out.is_null() { cudaFree(self.dev_shared_out as *mut c_void); }
            if !self.dev_exp_x.is_null() { cudaFree(self.dev_exp_x as *mut c_void); }
            if !self.dev_exp_gate.is_null() { cudaFree(self.dev_exp_gate as *mut c_void); }
            if !self.dev_exp_up.is_null() { cudaFree(self.dev_exp_up as *mut c_void); }
            if !self.dev_exp_act.is_null() { cudaFree(self.dev_exp_act as *mut c_void); }
            if !self.dev_exp_out.is_null() { cudaFree(self.dev_exp_out as *mut c_void); }
            if !self.dev_exp_dact.is_null() { cudaFree(self.dev_exp_dact as *mut c_void); }
            if !self.dev_exp_dgate.is_null() { cudaFree(self.dev_exp_dgate as *mut c_void); }
            if !self.dev_exp_dup.is_null() { cudaFree(self.dev_exp_dup as *mut c_void); }
            if !self.dev_exp_dx.is_null() { cudaFree(self.dev_exp_dx as *mut c_void); }
            if !self.dev_exp_dx_tmp.is_null() { cudaFree(self.dev_exp_dx_tmp as *mut c_void); }
            if !self.dev_exp_dy.is_null() { cudaFree(self.dev_exp_dy as *mut c_void); }
            if !self.dev_exp_indices.is_null() { cudaFree(self.dev_exp_indices as *mut c_void); }
            if !self.dev_exp_weights.is_null() { cudaFree(self.dev_exp_weights as *mut c_void); }
            if !self.dev_exp_scalars.is_null() { cudaFree(self.dev_exp_scalars as *mut c_void); }
            if !self.dev_rstd.is_null() { cudaFree(self.dev_rstd as *mut c_void); }

            if !self.cublas_handle.is_null() { cublasDestroy_v2(self.cublas_handle); }
        }
    }
}

static GPU_MODEL_ENGINE: Lazy<Mutex<Option<GpuModelEngine>>> = Lazy::new(|| Mutex::new(None));

pub fn init_gpu_model_engine(param_shapes: &[(usize, usize, bool)]) -> Result<()> {
    let mut lock = GPU_MODEL_ENGINE.lock().unwrap();
    *lock = Some(GpuModelEngine::new(param_shapes)?);
    Ok(())
}

pub fn with_gpu_model_engine<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut GpuModelEngine) -> R,
{
    let mut lock = GPU_MODEL_ENGINE.lock().unwrap();
    lock.as_mut().map(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_persistent_vram_gemm_correctness() {
        let shapes = vec![(4, 3, true)];
        let mut engine = GpuModelEngine::new(&shapes).expect("Failed to create GpuModelEngine");

        let w_host = vec![
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ];
        engine.upload_weights(&[&w_host]).expect("upload failed");

        let x_host = vec![
            1.0f32, 0.5, 0.2, 0.1,
            2.0, 1.0, 0.0, -1.0,
        ];
        let mut out_gpu = vec![0.0f32; 2 * 3];
        engine.forward_gemm(0, &x_host, &mut out_gpu, 2, 4, 3).expect("forward failed");

        let mut out_cpu = vec![0.0f32; 2 * 3];
        for i in 0..2 {
            for j in 0..3 {
                let mut sum = 0.0f32;
                for k in 0..4 {
                    sum += x_host[i * 4 + k] * w_host[k * 3 + j];
                }
                out_cpu[i * 3 + j] = sum;
            }
        }

        for i in 0..out_cpu.len() {
            let diff = (out_gpu[i] - out_cpu[i]).abs();
            assert!(diff < 1e-4, "Forward mismatch at {}: gpu {} vs cpu {}", i, out_gpu[i], out_cpu[i]);
        }

        let dy_host = vec![
            0.5f32, -0.5, 1.0,
            1.0, 2.0, -1.0,
        ];
        let mut dx_gpu = vec![0.0f32; 2 * 4];
        engine.backward_data_gemm(0, &dy_host, &mut dx_gpu, 2, 4, 3).expect("backward data failed");

        let mut dx_cpu = vec![0.0f32; 2 * 4];
        for i in 0..2 {
            for k in 0..4 {
                let mut sum = 0.0f32;
                for j in 0..3 {
                    sum += dy_host[i * 3 + j] * w_host[k * 3 + j];
                }
                dx_cpu[i * 4 + k] = sum;
            }
        }

        for i in 0..dx_cpu.len() {
            let diff = (dx_gpu[i] - dx_cpu[i]).abs();
            assert!(diff < 1e-4, "Backward data mismatch at {}: gpu {} vs cpu {}", i, dx_gpu[i], dx_cpu[i]);
        }

        engine.zero_grads().expect("zero grads failed");
        engine.backward_weight_gemm(0, &x_host, &dy_host, 2, 4, 3).expect("backward weight failed");

        engine.adamw_step(0.01, 0.9, 0.95, 1e-8, 0.0, 1).expect("adamw failed");

        let mut w_updated = vec![0.0f32; 12];
        engine.download_weights(&mut [&mut w_updated]).expect("download failed");

        assert_ne!(w_updated, w_host, "Weights must be updated by AdamW");
        println!("GPU VRAM GEMM & AdamW test passed successfully!");
    }

    #[test]
    fn test_gpu_batched_attention_execution() {
        let seq_len = 4usize;
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let head_dim = 4usize;
        let hidden_dim = num_heads * head_dim; // 8
        let kv_dim = num_kv_heads * head_dim; // 4

        let shapes = vec![
            (hidden_dim, hidden_dim, true), // Wq: 0
            (hidden_dim, kv_dim, true),     // Wk: 1
            (hidden_dim, kv_dim, true),     // Wv: 2
            (hidden_dim, hidden_dim, true), // Wo: 3
        ];
        let mut engine = GpuModelEngine::new(&shapes).expect("Failed to create engine");

        let wq = vec![0.1f32; hidden_dim * hidden_dim];
        let wk = vec![0.1f32; hidden_dim * kv_dim];
        let wv = vec![0.1f32; hidden_dim * kv_dim];
        let wo = vec![0.1f32; hidden_dim * hidden_dim];
        engine.upload_weights(&[&wq, &wk, &wv, &wo]).expect("upload failed");

        let cos_cache = vec![1.0f32; seq_len * (head_dim / 2)];
        let sin_cache = vec![0.0f32; seq_len * (head_dim / 2)];
        engine.upload_rope_cache(&cos_cache, &sin_cache).expect("rope upload failed");

        let normed1 = vec![0.5f32; seq_len * hidden_dim];
        let mut attn_out = vec![0.0f32; seq_len * hidden_dim];

        let (q_rope, k_rope, v, probs, concat) = engine.gpu_batched_attention_forward(
            &normed1,
            0, 1, 2, 3,
            &mut attn_out,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
        ).expect("GPU attention forward failed");

        assert_eq!(q_rope.len(), seq_len * hidden_dim);
        assert_eq!(k_rope.len(), seq_len * kv_dim);
        assert_eq!(v.len(), seq_len * kv_dim);
        assert_eq!(probs.len(), num_heads * seq_len * seq_len);
        assert_eq!(concat.len(), seq_len * hidden_dim);

        // Verify probabilities sum to ~1.0 for each token and head
        for h in 0..num_heads {
            for i in 0..seq_len {
                let row = &probs[h * seq_len * seq_len + i * seq_len..h * seq_len * seq_len + (i + 1) * seq_len];
                let sum: f32 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-4, "Softmax sum mismatch at h={}, i={}: {}", h, i, sum);
                // Causal check: future tokens must be 0
                for j in (i + 1)..seq_len {
                    assert_eq!(row[j], 0.0f32, "Causal violation at h={}, i={}, j={}", h, i, j);
                }
            }
        }

        // Test Backward
        let d_x_res = vec![0.1f32; seq_len * hidden_dim];
        let mut d_normed1 = vec![0.0f32; seq_len * hidden_dim];

        engine.gpu_batched_attention_backward(
            &d_x_res,
            &normed1,
            &q_rope,
            &k_rope,
            &v,
            &probs,
            &concat,
            0, 1, 2, 3,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            &mut d_normed1,
        ).expect("GPU attention backward failed");

        let has_nonzero = d_normed1.iter().any(|&x| x.abs() > 1e-6);
        assert!(has_nonzero, "d_normed1 must contain non-zero gradients!");
        println!("✅ [TEST PASS] GPU Batched Attention Forward & Backward verified with causal mask & non-zero gradients!");
    }

    #[test]
    fn test_gpu_rmsnorm_correctness() {
        let seq_len = 2usize;
        let hidden_dim = 4usize;
        let shapes = vec![(1, hidden_dim, false)]; // weight at param 0
        let mut engine = GpuModelEngine::new(&shapes).expect("Failed to create engine");

        let weight = vec![1.0f32, 1.0, 1.0, 1.0];
        engine.upload_weights(&[&weight]).expect("upload failed");

        let x = vec![
            1.0f32, 2.0, 3.0, 4.0,
            2.0, 2.0, 2.0, 2.0,
        ];
        let mut out = vec![0.0f32; seq_len * hidden_dim];
        let mut rstd = vec![0.0f32; seq_len];

        engine.gpu_rmsnorm_forward(&x, 0, &mut out, &mut rstd, seq_len, hidden_dim, 1e-5).expect("rmsnorm fwd failed");

        // Verify row 1: mean_sq = (4+4+4+4)/4 = 4. rstd = 1/sqrt(4) = 0.5. out = 2 * 0.5 * 1 = 1.0!
        for d in 4..8 {
            assert!((out[d] - 1.0).abs() < 1e-4, "Expected 1.0, got {}", out[d]);
        }

        // Test backward
        let dout = vec![0.1f32; seq_len * hidden_dim];
        let mut dx = vec![0.0f32; seq_len * hidden_dim];
        engine.zero_grads().expect("zero grads failed");
        engine.gpu_rmsnorm_backward(&dout, &x, &rstd, 0, &mut dx, seq_len, hidden_dim).expect("rmsnorm bwd failed");

        let has_nonzero = dx.iter().any(|&v| v.abs() > 1e-6);
        assert!(has_nonzero, "RMSNorm dx must have non-zero elements");
        println!("✅ [TEST PASS] GPU RMSNorm Forward & Backward verified!");
    }

    #[test]
    fn test_gpu_moe_forward_backward() {
        let seq_len = 2usize;
        let hidden_dim = 4usize;
        let intermediate_dim = 8usize;
        let num_experts = 2usize;

        // Shared: gate (0), up (1), down (2)
        // Expert 0: gate (3), up (4), down (5)
        // Expert 1: gate (6), up (7), down (8)
        // Router: (9)
        let shapes = vec![
            (hidden_dim, intermediate_dim, true), // 0: shared_gate
            (hidden_dim, intermediate_dim, true), // 1: shared_up
            (intermediate_dim, hidden_dim, true), // 2: shared_down
            (hidden_dim, intermediate_dim, true), // 3: exp0_gate
            (hidden_dim, intermediate_dim, true), // 4: exp0_up
            (intermediate_dim, hidden_dim, true), // 5: exp0_down
            (hidden_dim, intermediate_dim, true), // 6: exp1_gate
            (hidden_dim, intermediate_dim, true), // 7: exp1_up
            (intermediate_dim, hidden_dim, true), // 8: exp1_down
            (hidden_dim, num_experts, true),      // 9: router
        ];
        let mut engine = GpuModelEngine::new(&shapes).expect("Failed to create engine");

        let w_gate = vec![0.1f32; hidden_dim * intermediate_dim];
        let w_up = vec![0.1f32; hidden_dim * intermediate_dim];
        let w_down = vec![0.1f32; intermediate_dim * hidden_dim];
        let w_router = vec![0.1f32; hidden_dim * num_experts];

        engine.upload_weights(&[
            &w_gate, &w_up, &w_down,
            &w_gate, &w_up, &w_down,
            &w_gate, &w_up, &w_down,
            &w_router,
        ]).expect("upload failed");

        let normed2 = vec![0.5f32; seq_len * hidden_dim];

        // 1. Forward Shared
        let (s_gate, s_up, s_act, s_out) = engine.gpu_moe_forward_shared(
            &normed2,
            0, 1, 2,
            seq_len,
            hidden_dim,
            intermediate_dim,
        ).expect("shared fwd failed");

        assert_eq!(s_out.len(), seq_len * hidden_dim);

        // 2. Forward Expert 0
        let token_indices = vec![0i32, 1i32];
        let weights = vec![0.5f32, 0.5f32];
        let (e0_x, e0_gate, e0_up, e0_act, e0_out) = engine.gpu_moe_forward_expert(
            &token_indices,
            &weights,
            3, 4, 5,
            hidden_dim,
            intermediate_dim,
        ).expect("exp0 fwd failed");

        assert_eq!(e0_out.len(), seq_len * hidden_dim);

        // 3. Finalize forward
        let mut total_moe_out = vec![0.0f32; seq_len * hidden_dim];
        engine.gpu_moe_finalize_forward(&mut total_moe_out).expect("finalize fwd failed");

        let has_nonzero = total_moe_out.iter().any(|&v| v.abs() > 1e-6);
        assert!(has_nonzero, "MoE forward output must be non-zero");

        // 4. Backward
        let d_moe = vec![0.1f32; seq_len * hidden_dim];
        engine.zero_grads().expect("zero grads failed");
        engine.gpu_moe_init_backward(&d_moe).expect("init bwd failed");

        let scalars = engine.gpu_moe_backward_expert(
            &token_indices,
            &weights,
            &e0_x,
            &e0_gate,
            &e0_up,
            &e0_act,
            &e0_out,
            3, 4, 5,
            hidden_dim,
            intermediate_dim,
        ).expect("exp0 bwd failed");

        assert_eq!(scalars.len(), seq_len);

        engine.gpu_moe_backward_shared(
            &s_act,
            &s_gate,
            &s_up,
            &normed2,
            0, 1, 2,
            seq_len,
            hidden_dim,
            intermediate_dim,
        ).expect("shared bwd failed");

        let d_router_logits = vec![0.05f32; seq_len * num_experts];
        let mut d_normed2 = vec![0.0f32; seq_len * hidden_dim];
        engine.gpu_moe_backward_router(
            &normed2,
            &d_router_logits,
            9,
            seq_len,
            hidden_dim,
            num_experts,
            &mut d_normed2,
        ).expect("router bwd failed");

        let has_nonzero_grad = d_normed2.iter().any(|&v| v.abs() > 1e-6);
        assert!(has_nonzero_grad, "d_normed2 must have non-zero gradients!");
        println!("✅ [TEST PASS] GPU MoE Forward & Backward verified with non-zero gradients!");
    }

    #[test]
    fn test_gpu_clip_grad_norm() {
        let shapes = vec![(2, 2, true)]; // 4 parameters
        let mut engine = GpuModelEngine::new(&shapes).expect("Failed to create engine");

        // Manually upload gradient vector [3.0, 4.0, 0.0, 0.0] -> L2 norm = sqrt(9 + 16) = 5.0
        let g_host = vec![3.0f32, 4.0, 0.0, 0.0];
        unsafe {
            cudaMemcpy(
                engine.dev_grads as *mut c_void,
                g_host.as_ptr() as *const c_void,
                g_host.len() * std::mem::size_of::<f32>(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );
        }

        let initial_norm = engine.clip_grad_norm(1.0).expect("clip failed");
        assert!((initial_norm - 5.0).abs() < 1e-4, "Initial norm must be 5.0, got {}", initial_norm);

        // After clipping with max_norm=1.0, norm must now be 1.0
        let clipped_norm = engine.clip_grad_norm(10.0).expect("measure norm failed");
        assert!((clipped_norm - 1.0).abs() < 1e-4, "Clipped norm must be 1.0, got {}", clipped_norm);
        println!("✅ [TEST PASS] GPU Hardware Gradient Clipping (cuBLAS Snrm2 + Sscal) verified!");
    }
}
