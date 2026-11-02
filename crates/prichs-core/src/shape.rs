#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn elem_count(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn default_strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.dims.len()];
        for i in (0..self.dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }
}

impl From<&[usize]> for Shape {
    fn from(s: &[usize]) -> Self {
        Self::new(s.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(s: [usize; N]) -> Self {
        Self::new(s.to_vec())
    }
}
