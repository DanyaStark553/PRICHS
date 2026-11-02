use crate::dtype::DType;
use half::{bf16, f16};

#[derive(Debug, Clone)]
pub enum CpuStorage {
    F32(Vec<f32>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
}

impl CpuStorage {
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F16(_) => DType::F16,
            Self::BF16(_) => DType::BF16,
            Self::U32(_) => DType::U32,
            Self::I32(_) => DType::I32,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F16(v) => v.len(),
            Self::BF16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f32_slice(&self) -> anyhow::Result<&[f32]> {
        match self {
            Self::F32(v) => Ok(v.as_slice()),
            _ => anyhow::bail!("Expected F32 storage"),
        }
    }

    pub fn as_f32_mut_slice(&mut self) -> anyhow::Result<&mut [f32]> {
        match self {
            Self::F32(v) => Ok(v.as_mut_slice()),
            _ => anyhow::bail!("Expected F32 storage"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Storage {
    Cpu(CpuStorage),
    // CUDA storage will be plugged in via prichs-cuda
}

impl Storage {
    pub fn dtype(&self) -> DType {
        match self {
            Self::Cpu(cpu) => cpu.dtype(),
        }
    }
}
