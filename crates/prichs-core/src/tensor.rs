use crate::dtype::DType;
use crate::shape::Shape;
use crate::storage::{CpuStorage, Storage};
use anyhow::{bail, Result};

#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Storage,
    shape: Shape,
    pub strides: Vec<usize>,
}

impl Tensor {
    pub fn from_f32_slice(data: &[f32], shape: impl Into<Shape>) -> Result<Self> {
        let shape: Shape = shape.into();
        if data.len() != shape.elem_count() {
            bail!(
                "Data length {} does not match shape count {}",
                data.len(),
                shape.elem_count()
            );
        }
        let strides = shape.default_strides();
        Ok(Self {
            storage: Storage::Cpu(CpuStorage::F32(data.to_vec())),
            shape,
            strides,
        })
    }

    pub fn zeros(shape: impl Into<Shape>, dtype: DType) -> Result<Self> {
        let shape: Shape = shape.into();
        let count = shape.elem_count();
        let storage = match dtype {
            DType::F32 => Storage::Cpu(CpuStorage::F32(vec![0.0; count])),
            DType::F16 => Storage::Cpu(CpuStorage::F16(vec![half::f16::ZERO; count])),
            DType::BF16 => Storage::Cpu(CpuStorage::BF16(vec![half::bf16::ZERO; count])),
            DType::U32 => Storage::Cpu(CpuStorage::U32(vec![0; count])),
            DType::I32 => Storage::Cpu(CpuStorage::I32(vec![0; count])),
        };
        let strides = shape.default_strides();
        Ok(Self {
            storage,
            shape,
            strides,
        })
    }

    pub fn randn(shape: impl Into<Shape>, mean: f32, std: f32) -> Result<Self> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let shape: Shape = shape.into();
        let count = shape.elem_count();
        let mut data = Vec::with_capacity(count);
        for _ in 0..count {
            // Box-Muller transform for normal distribution
            let u1: f32 = rng.gen::<f32>().max(1e-7);
            let u2: f32 = rng.gen::<f32>();
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            data.push(mean + z0 * std);
        }
        Self::from_f32_slice(&data, shape)
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn storage_mut(&mut self) -> &mut Storage {
        &mut self.storage
    }

    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        match &self.storage {
            Storage::Cpu(cpu) => cpu.as_f32_slice(),
        }
    }

    pub fn as_f32_mut_slice(&mut self) -> Result<&mut [f32]> {
        match &mut self.storage {
            Storage::Cpu(cpu) => cpu.as_f32_mut_slice(),
        }
    }
}
