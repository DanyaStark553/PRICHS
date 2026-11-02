#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    U32,
    I32,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}
