use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::dataloader::{DataError, PrichsMmapDataset};
use crate::schedule::CosineScheduler;

#[derive(Debug, Clone)]
pub struct TrainerConfig {
    pub micro_batch_size: usize,
    pub seq_len: usize,
    pub grad_accum_steps: usize,
    pub max_steps: usize,
    pub max_grad_norm: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub router_bias_gamma: f32,
    pub eval_interval: usize,
    pub save_dir: PathBuf,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            micro_batch_size: 4,
            seq_len: 1024, // Baby-П.Р.І.Ч.С. на RTX 3060 (Ефективний батч = 4 * 8 * 1024 = 32,768 токенів/крок)
            grad_accum_steps: 8,
            max_steps: 10_000,
            max_grad_norm: 1.0,
            weight_decay: 0.1,
            beta1: 0.9,
            beta2: 0.95, // 0.95 замість 0.999 критично для стабільності LLM при спалахах градієнтів
            eps: 1e-8,
            router_bias_gamma: 0.001,
            eval_interval: 250,
            save_dir: PathBuf::from("checkpoints"),
        }
    }
}

pub struct ParameterGroup {
    pub data: Vec<f32>,           // FP32 Master Weights
    pub grad: Vec<f32>,           // Накопичені градієнти за grad_accum_steps
    pub exp_avg: Vec<f32>,        // Adam m (Перший момент)
    pub exp_avg_sq: Vec<f32>,     // Adam v (Другий момент)
    pub apply_weight_decay: bool, // true для 2D матриць, false для RMSNorm та Router Gate
}

pub struct AdamWOptimizer {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step_count: usize,
}

impl AdamWOptimizer {
    pub fn new(beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            beta1,
            beta2,
            eps,
            weight_decay,
            step_count: 0,
        }
    }

    /// Global L2 Norm Clipping по всіх параметрах моделі перед кроком оптимізатора
    pub fn clip_grad_norm(groups: &mut [ParameterGroup], max_norm: f32) -> f32 {
        let mut total_norm_sq = 0.0f64;
        for group in groups.iter() {
            for &g in group.grad.iter() {
                let g64 = g as f64;
                total_norm_sq += g64 * g64;
            }
        }
        let total_norm = (total_norm_sq as f32).sqrt();
        if total_norm > max_norm {
            let scale = max_norm / (total_norm + 1e-6);
            for group in groups.iter_mut() {
                for g in group.grad.iter_mut() {
                    *g *= scale;
                }
            }
        }
        total_norm
    }

    pub fn step(&mut self, groups: &mut [ParameterGroup], lr: f32) {
        self.step_count += 1;
        let t = self.step_count as i32;
        let bias_correction1 = 1.0 - self.beta1.powi(t);
        let bias_correction2 = 1.0 - self.beta2.powi(t);
        let step_size = lr / bias_correction1;

        for group in groups.iter_mut() {
            let wd = if group.apply_weight_decay {
                self.weight_decay
            } else {
                0.0
            };

            for i in 0..group.data.len() {
                let g = group.grad[i];
                let mut w = group.data[i];

                // Decoupled Weight Decay (виконується ДО оновлення моментуму)
                if wd > 0.0 {
                    w -= lr * wd * w;
                }

                let m = self.beta1 * group.exp_avg[i] + (1.0 - self.beta1) * g;
                let v = self.beta2 * group.exp_avg_sq[i] + (1.0 - self.beta2) * (g * g);

                group.exp_avg[i] = m;
                group.exp_avg_sq[i] = v;

                let denom = (v / bias_correction2).sqrt() + self.eps;
                group.data[i] = w - step_size * (m / denom);

                // Онулення градієнта одразу в тій самій кеш-лінії (Fused Zero-Grad)
                group.grad[i] = 0.0;
            }
        }
    }
}

/// Трейт для інтеграції моделі Prichs з тренувальним ядром
pub trait TrainableMoEModel {
    fn forward_backward_microbatch(
        &mut self,
        inputs: &[u16],
        targets: &[u16],
        cu_seqlens: &[i32],
        loss_scale: f32,
    ) -> f32;

    fn update_router_biases(&mut self, gamma: f32);
    fn parameter_groups_mut(&mut self) -> &mut [ParameterGroup];
    fn sync_master_to_bf16_weights(&mut self);
}

pub struct PrichsTrainer {
    pub config: TrainerConfig,
    pub optimizer: AdamWOptimizer,
    pub scheduler: CosineScheduler,
    pub best_loss: f32,
}

impl PrichsTrainer {
    pub fn new(config: TrainerConfig, scheduler: CosineScheduler) -> std::io::Result<Self> {
        fs::create_dir_all(&config.save_dir)?;
        let optimizer = AdamWOptimizer::new(
            config.beta1,
            config.beta2,
            config.eps,
            config.weight_decay,
        );
        Ok(Self {
            config,
            optimizer,
            scheduler,
            best_loss: f32::INFINITY,
        })
    }

    pub fn train_loop<M: TrainableMoEModel>(
        &mut self,
        model: &mut M,
        dataset: &PrichsMmapDataset,
    ) -> Result<(), DataError> {
        let tokens_per_micro = self.config.micro_batch_size * self.config.seq_len;
        let tokens_per_step = tokens_per_micro * self.config.grad_accum_steps;
        let loss_scale = 1.0 / (self.config.grad_accum_steps as f32);

        let mut data_cursor: usize = 0;
        let max_tokens = dataset.header.total_tokens as usize;

        for step in 0..self.config.max_steps {
            let step_start = Instant::now();
            let mut step_loss = 0.0f32;

            // 1. Цикл накопичення градієнтів (Gradient Accumulation)
            for _micro in 0..self.config.grad_accum_steps {
                if data_cursor + tokens_per_micro + 1 >= max_tokens {
                    data_cursor = 0; // Початок нової епохи по шарду
                }

                let (x, y, cu_seqlens) = dataset.get_sequence_with_boundaries(
                    data_cursor,
                    tokens_per_micro,
                )?;
                data_cursor += tokens_per_micro;

                // Forward + Backward з попереднім масштабуванням на 1 / grad_accum_steps
                let micro_loss = model.forward_backward_microbatch(x, y, &cu_seqlens, loss_scale);
                step_loss += micro_loss * loss_scale;
            }

            // 2. Відсікання градієнтів (Global L2 Clipping <= 1.0)
            let grad_norm = AdamWOptimizer::clip_grad_norm(
                model.parameter_groups_mut(),
                self.config.max_grad_norm,
            );

            // 3. Отримання актуального LR та крок оптимізатора (з Fused Zero-Grad)
            let lr = self.scheduler.get_lr(step);
            self.optimizer.step(model.parameter_groups_mut(), lr);

            // 4. Синхронізація оновлених FP32 Master ваг у робочі BF16 тензори
            model.sync_master_to_bf16_weights();

            // 5. DeepSeek-V3 Auxiliary-Loss-Free оновлення зміщень роутера раз на крок
            model.update_router_biases(self.config.router_bias_gamma);

            let elapsed_sec = step_start.elapsed().as_secs_f32();
            let tok_per_sec = (tokens_per_step as f32) / elapsed_sec;

            if step % 10 == 0 || step == self.config.max_steps - 1 {
                println!(
                    "⚡ [Крок {:05}/{}] Loss: {:.4} | GradNorm: {:.3} | LR: {:.6} | Speed: {:.0} tok/s",
                    step, self.config.max_steps, step_loss, grad_norm, lr, tok_per_sec
                );
            }

            // 6. Автоматичний чекпоінтинг найкращої моделі (.prichs)
            if step > 0 && step % self.config.eval_interval == 0 && step_loss < self.best_loss {
                self.best_loss = step_loss;
                let ckpt_path = self.config.save_dir.join("baby_prichs_best.prichs");
                self.save_checkpoint(&ckpt_path, step, step_loss, model.parameter_groups_mut())?;
                println!("💾 [Чекпоінт] Новий рекорд Loss ({:.4}) збережено у {:?}", step_loss, ckpt_path);
            }
        }

        Ok(())
    }

    /// Атомарне збереження ваг у нативному форматі .prichs (Write-to-Temp + Rename)
    pub fn save_checkpoint(
        &self,
        path: &Path,
        step: usize,
        loss: f32,
        groups: &[ParameterGroup],
    ) -> std::io::Result<()> {
        let tmp_path = path.with_extension("prichs.tmp");
        {
            let file = File::create(&tmp_path)?;
            let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file);

            writer.write_all(b"PRICHS_W")?;
            writer.write_all(&(step as u64).to_le_bytes())?;
            writer.write_all(&loss.to_le_bytes())?;
            writer.write_all(&(groups.len() as u32).to_le_bytes())?;

            for group in groups {
                let byte_len = (group.data.len() * 4) as u64;
                writer.write_all(&byte_len.to_le_bytes())?;
                let raw_bytes = unsafe {
                    std::slice::from_raw_parts(group.data.as_ptr() as *const u8, byte_len as usize)
                };
                writer.write_all(raw_bytes)?;
            }
            writer.flush()?;
        }
        // Атомарний POSIX rename гарантує, що чекпоінт не поб'ється при раптовому вимкненні живлення
        fs::rename(tmp_path, path)?;
        Ok(())
    }
}
