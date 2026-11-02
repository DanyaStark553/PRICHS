pub mod bridge;
pub mod cublas;
pub mod engine;
pub mod ipc;

pub use bridge::{MysteriumReadOnlyBridge, StarkSynapseSlice};
pub use cublas::{
    get_gpu_workspace, matmul_a_bt_gpu, matmul_at_b_gpu, matmul_gpu, GpuWorkspace,
};
pub use engine::{init_gpu_model_engine, with_gpu_model_engine, GpuModelEngine, ParamMetadata};
pub use ipc::{CudaIpcChannel, MemoryHandleDescriptor};

pub struct CudaDevice {
    pub device_id: usize,
}

impl CudaDevice {
    pub fn new(device_id: usize) -> Self {
        Self { device_id }
    }

    pub fn is_available() -> bool {
        get_gpu_workspace().is_some()
    }
}
