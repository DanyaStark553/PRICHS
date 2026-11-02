use std::f32::consts::PI;

pub type CosineScheduler = CosineLrSchedule;

/// Cosine Annealing Learning Rate Schedule with Linear Warmup
#[derive(Debug, Clone)]
pub struct CosineLrSchedule {
    pub max_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl CosineLrSchedule {
    pub fn new(max_lr: f32, min_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            max_lr,
            min_lr,
            warmup_steps,
            total_steps,
        }
    }

    pub fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup: from min_lr to max_lr
            let alpha = step as f32 / self.warmup_steps.max(1) as f32;
            self.min_lr + alpha * (self.max_lr - self.min_lr)
        } else if step >= self.total_steps {
            self.min_lr
        } else {
            // Cosine decay
            let progress = (step - self.warmup_steps) as f32 / (self.total_steps - self.warmup_steps) as f32;
            let cosine = (1.0 + (progress * PI).cos()) * 0.5;
            self.min_lr + cosine * (self.max_lr - self.min_lr)
        }
    }
}
