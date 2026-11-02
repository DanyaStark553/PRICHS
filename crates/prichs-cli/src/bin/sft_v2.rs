use anyhow::{bail, Context, Result};
use prichs_cuda::GpuModelEngine;
use prichs_model::{PrichsConfig, PrichsModel};
use prichs_tokenizer::PrichsTokenizer;
use prichs_train::PrichsSftDataset;
use std::collections::VecDeque;
use std::path::Path;
use std::time::Instant;

#[cfg(windows)]
fn get_process_memory_mb() -> (f64, f64) {
    #[repr(C)]
    struct PROCESS_MEMORY_COUNTERS {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(
            process: isize,
            ppsmc: *mut PROCESS_MEMORY_COUNTERS,
            cb: u32,
        ) -> i32;
    }
    unsafe {
        let mut pmc: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        pmc.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        let proc = GetCurrentProcess();
        if K32GetProcessMemoryInfo(proc, &mut pmc, pmc.cb) != 0 {
            (
                pmc.working_set_size as f64 / (1024.0 * 1024.0),
                pmc.pagefile_usage as f64 / (1024.0 * 1024.0),
            )
        } else {
            (0.0, 0.0)
        }
    }
}

#[cfg(not(windows))]
fn get_process_memory_mb() -> (f64, f64) {
    (0.0, 0.0)
}

fn count_parameters(model: &PrichsModel) -> (usize, usize) {
    let params = model.get_parameter_tensors();
    let total: usize = params.iter().map(|(t, _)| t.dims().iter().product::<usize>()).sum();

    let cfg = &model.config;
    let wte_params = cfg.vocab_size * cfg.hidden_dim;
    let final_norm_params = cfg.hidden_dim;

    let attn_params = cfg.hidden_dim * cfg.hidden_dim
        + cfg.hidden_dim * (cfg.num_kv_heads * (cfg.hidden_dim / cfg.num_attention_heads))
        + cfg.hidden_dim * (cfg.num_kv_heads * (cfg.hidden_dim / cfg.num_attention_heads))
        + cfg.hidden_dim * cfg.hidden_dim;

    let norms_params = cfg.hidden_dim * 2;
    let router_params = cfg.hidden_dim * cfg.num_experts;
    let expert_params = 3 * cfg.hidden_dim * cfg.intermediate_dim;
    let active_moe_params = router_params
        + expert_params * cfg.num_shared_experts
        + expert_params * cfg.num_experts_per_tok;

    let active_per_layer = attn_params + norms_params + active_moe_params;
    let active = wte_params + cfg.num_layers * active_per_layer + final_norm_params;

    (total, active)
}

fn compute_lr(step: usize, max_steps: usize, warmup_steps: usize, max_lr: f32, min_lr: f32) -> f32 {
    if step < warmup_steps {
        max_lr * (step + 1) as f32 / warmup_steps as f32
    } else if step >= max_steps {
        min_lr
    } else {
        let progress = (step - warmup_steps) as f32 / (max_steps - warmup_steps) as f32;
        let cosine_decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        min_lr + (max_lr - min_lr) * cosine_decay
    }
}

fn generate_shuffled_samples(num_samples: usize, epoch_seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..num_samples).collect();
    let mut state = epoch_seed ^ 0x9e3779b97f4a7c15;
    if state == 0 {
        state = 1;
    }
    for i in (1..indices.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        indices.swap(i, j);
    }
    indices
}

fn main() -> Result<()> {
    println!("\x1b[38;2;244;63;94m");
    println!(r#"
  ██████╗         ██████╗       ██╗       ██████╗     ██╗  ██╗
  ██╔══██╗        ██╔══██╗      ██║      ██╔════╝     ██║  ██║
  ██████╔╝  ██╗   ██████╔╝  ██╗ ██║  ██╗ ██║    ██╗   ███████║
  ██╔═══╝   ╚═╝   ██╔══██╗  ╚═╝ ██║  ╚═╝ ██║    ╚═╝   ██╔══██║
  ██║             ██║  ██║      ██║      ╚██████╗     ██║  ██║
  ╚═╝             ╚═╝  ╚═╝      ╚═╝       ╚═════╝     ╚═╝  ╚═╝
  П. Р. І. Ч. С. v2 // Sovereign SFT Character & Alignment Engine
  Masked Cross-Entropy Loss // Zero-Copy Isolated Context Pipeline
    "#);
    println!("\x1b[0m");

    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("v2");
    let max_steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3500);
    let accum_steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);

    let (cfg, model_title, max_lr, min_lr) = match mode {
        "baby" => {
            (PrichsConfig::baby(), "Baby-П.Р.І.Ч.С. SFT (52M Total / 19M Active)", 0.000050f32, 0.000005f32)
        }
        "mid" | "355m" => {
            let mut c = PrichsConfig::v2_moe();
            c.num_layers = 12;
            (c, "П.Р.І.Ч.С. v2 Balanced SFT (12L / 355M Total / 121M Active)", 0.000035f32, 0.000003f32)
        }
        "v2" | "472m" => {
            (PrichsConfig::v2_moe(), "П.Р.І.Ч.С. v2 Heavy MoE SFT (16L / 472M Total / 160M Active)", 0.000080f32, 0.000010f32)
        }
        other => {
            println!("Невідомий профіль: '{}'. Допустимі: 'v2', 'mid', 'baby'.", other);
            std::process::exit(1);
        }
    };

    // Визначаємо шлях до датасету SFT
    let explicit_dataset = args.iter().find(|a| a.ends_with(".prichsbin")).cloned();
    let default_sft_dataset = "data/sft_v2.prichsbin";
    let dataset_path_str = explicit_dataset.unwrap_or_else(|| default_sft_dataset.to_string());
    let dataset_path = Path::new(&dataset_path_str);

    if !dataset_path.exists() {
        bail!("Файл SFT датасету {:?} не знайдено! Переконайтеся, що запущено compile_sft_v2.py.", dataset_path);
    }

    let sft_dataset = PrichsSftDataset::open(dataset_path)
        .context("Не вдалося відкрити бінарний SFT датасет")?;

    let num_samples = sft_dataset.num_samples();
    let seq_len = sft_dataset.seq_len();
    let steps_per_epoch = num_samples / accum_steps;
    let total_epochs_target = (max_steps as f64) / (steps_per_epoch as f64);

    let tokenizer_path = Path::new("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        println!("📖 [TOKENIZER] Завантажено токенізатор із {:?}", tokenizer_path);
        Some(PrichsTokenizer::from_huggingface_json(tokenizer_path)?)
    } else {
        println!("⚠️ [TOKENIZER] tokenizer.json не знайдено. Live Monitor вимкнено.");
        None
    };

    println!("========================================================");
    println!("  🎭 ПРОФІЛЬ SFT ФАЙНТЮНІНГУ: {}", model_title);
    println!("  SFT Датасет:      {} ({} унікальних діалогів)", dataset_path.display(), num_samples);
    println!("  Контекст T:       {} токенів (Ізольовані вікна)", seq_len);
    println!("  Діалогів на крок: {} діалогів (Grad Accum = micro-batches)", accum_steps);
    println!("  1 епоха:          {} кроків", steps_per_epoch);
    println!("  Цільових кроків:  {} ({:.2} повних епох)", max_steps, total_epochs_target);
    println!("  LR розклад:       {:.6} -> {:.6} (Cosine Decay, 50 кроків warmup)", max_lr, min_lr);
    println!("  Loss Функція:     Masked Cross-Entropy (prompt & pad mask = 0)");
    println!("  Gradient Clip:    max_grad_norm = 1.0 (Hardware GPU)");
    println!("========================================================\n");

    let (ram_init, _) = get_process_memory_mb();
    let (vram_free_start, vram_total) = GpuModelEngine::get_gpu_memory_info().unwrap_or((0, 0));
    println!("📊 Початковий стан хоста:");
    println!("   > Системна RAM процесу: {:.2} MB", ram_init);
    println!("   > Вільна VRAM GPU:      {:.2} GB із {:.2} GB",
        vram_free_start as f64 / (1024.0 * 1024.0 * 1024.0), vram_total as f64 / (1024.0 * 1024.0 * 1024.0));

    // Шукаємо базовий золотий чекпоінт для SFT
    let explicit_ckpt = args.iter().position(|a| a == "--checkpoint" || a == "-c")
        .and_then(|idx| args.get(idx + 1).cloned());

    let gold_ckpts = [
        "checkpoints/v2_sft_best.prichs",
        "checkpoints/v2_sft_final.prichs",
        "checkpoints/v2_pretrain_v2_step14951.prichs",
        "checkpoints/v2_pretrain_v2_final.prichs",
        "checkpoints/v2_pretrain_v2_step15000.prichs",
        "checkpoints/v2_pretrain_v2_best.prichs",
    ];

    let target_ckpt_path = if let Some(ref p) = explicit_ckpt {
        std::path::PathBuf::from(p)
    } else {
        let found = gold_ckpts.iter().find(|p| Path::new(p).exists());
        match found {
            Some(p) => std::path::PathBuf::from(p),
            None => bail!("Не знайдено базового чекпоінта пре-трейну (очікувався step 14951 або final у checkpoints/)!"),
        }
    };

    println!("\n⏳ [1/3] Завантаження золотих ваг моделі з {:?}...", target_ckpt_path);
    let t_model = Instant::now();
    let mut model = PrichsModel::new(cfg.clone())?;
    let (step_loaded, loss_loaded) = model.load_checkpoint(&target_ckpt_path)
        .context("Помилка завантаження чекпоінта пре-трейну")?;
    let (total_params, active_params) = count_parameters(&model);
    println!("   > Успішно відновлено ваги моделі (Пре-трейн крок: {}, Loss: {:.4})", step_loaded, loss_loaded);
    println!("   > Загальних параметрів: {:.2}M ({})", total_params as f64 / 1e6, total_params);
    println!("   > Активних на токен:    {:.2}M ({})", active_params as f64 / 1e6, active_params);
    println!("   > Час завантаження:     {:.2?}", t_model.elapsed());

    println!("\n⏳ [2/3] Алокація Persistent GPU VRAM Engine на RTX 3060...");
    let t_gpu = Instant::now();
    let mut engine = model.create_gpu_engine()?;
    let (vram_free_after, _) = GpuModelEngine::get_gpu_memory_info().unwrap_or((0, 0));
    let vram_used_mb = (vram_free_start.saturating_sub(vram_free_after)) as f64 / (1024.0 * 1024.0);
    println!("   > Час завантаження в VRAM: {:.2?}", t_gpu.elapsed());
    println!("   > Використано VRAM:        {:.2} MB ({:.2} GB)", vram_used_mb, vram_used_mb / 1024.0);
    println!("   > Залишок вільної VRAM:    {:.2} GB", vram_free_after as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("   > AdamW буфери m=0, v=0:   ІНІЦІАЛІЗОВАНО ЧИСТИМИ НУЛЯМИ ДЛЯ SFT");

    let warmup_steps = (max_steps / 20).max(5).min(50);
    let ckpt_dir = Path::new("checkpoints");
    std::fs::create_dir_all(ckpt_dir)?;
    let best_sft_path = format!("checkpoints/{}_sft_best.prichs", mode);

    let mut epoch = 0usize;
    let mut shuffled_samples = generate_shuffled_samples(num_samples, epoch as u64);
    let mut sample_cursor = 0usize;

    println!("\n🚀 [3/3] Запуск SFT файн-тюнінгу характеру у 12GB VRAM...");
    let train_start = Instant::now();
    let mut best_loss = f32::INFINITY;
    let mut best_rolling_loss = f32::INFINITY;
    let mut rolling_loss_window: VecDeque<f32> = VecDeque::with_capacity(50);

    for step in 0..max_steps {
        let step_start = Instant::now();
        let lr = compute_lr(step, max_steps, warmup_steps, max_lr, min_lr);

        let mut step_loss = 0.0f32;
        let grad_scale = 1.0f32 / (accum_steps as f32);

        // Hardware-accelerated persistent gradient zeroing ONCE per step
        engine.zero_grads()?;

        for _a in 0..accum_steps {
            if sample_cursor >= shuffled_samples.len() {
                epoch += 1;
                shuffled_samples = generate_shuffled_samples(num_samples, epoch as u64);
                sample_cursor = 0;
                println!("\n🔄 [SFT EPOCH] Перехід на епоху {} (Нове випадкове перемішування {} діалогів!)", epoch + 1, num_samples);
            }

            let s_idx = shuffled_samples[sample_cursor];
            sample_cursor += 1;

            let (inputs, targets, mask_u8) = sft_dataset.get_sample(s_idx)?;
            let mask_f32: Vec<f32> = mask_u8.iter().map(|&m| m as f32).collect();

            let micro_loss = model.forward_backward_vram_accum(&mut engine, inputs, targets, &mask_f32, grad_scale)?;
            step_loss += micro_loss / (accum_steps as f32);
        }

        // Hardware-accelerated Gradient Clipping in VRAM
        let grad_norm = engine.clip_grad_norm(1.0)?;

        // Fused AdamW Step directly on GPU VRAM
        engine.adamw_step(lr, 0.9, 0.95, 1e-8, 0.1, step + 1)?;

        // Track 50-step Rolling Average Loss
        rolling_loss_window.push_back(step_loss);
        if rolling_loss_window.len() > 50 {
            rolling_loss_window.pop_front();
        }
        let rolling_avg_loss: f32 = rolling_loss_window.iter().sum::<f32>() / (rolling_loss_window.len() as f32);

        let step_dur = step_start.elapsed();
        let step_tokens = seq_len * accum_steps;
        let step_tok_s = step_tokens as f64 / step_dur.as_secs_f64();

        let (ram_current, _) = get_process_memory_mb();
        let (vram_free_current, _) = GpuModelEngine::get_gpu_memory_info().unwrap_or((0, 0));

        let progress = (step + 1) as f64 / max_steps as f64;
        let remaining_steps = max_steps.saturating_sub(step + 1);
        let eta_secs = (remaining_steps as f64 * step_dur.as_secs_f64()) as u64;
        let eta_str = format!("{:02}:{:02}:{:02}", eta_secs / 3600, (eta_secs % 3600) / 60, eta_secs % 60);

        if step % 5 == 0 || step == max_steps - 1 {
            println!(
                "[{:04}/{:04}] ({:.1}%) | Ep:{:.2} | Loss: {:.4} (Roll50: {:.4}) | GradNorm: {:.2} | LR: {:.6} | Час: {:.2?} | {:.1} tok/s | RAM: {:.0} MB | VRAM: {:.2} GB | ETA: {}",
                step + 1, max_steps, progress * 100.0,
                (step + 1) as f64 / steps_per_epoch as f64,
                step_loss, rolling_avg_loss, grad_norm, lr, step_dur, step_tok_s, ram_current,
                (vram_total - vram_free_current) as f64 / (1024.0 * 1024.0 * 1024.0), eta_str
            );
        }

        // Live Emergence Monitor every 200 steps (and at step 100 and final step)
        if (step + 1) % 200 == 0 || (step + 1) == 100 || (step + 1) == max_steps {
            if let Some(ref tok) = tokenizer {
                model.sync_weights_from_engine(&engine)?;
                println!("\n🎭 [LIVE SFT CHARACTER MONITOR // Крок {} | Loss: {:.4} | Roll50: {:.4} | GradNorm: {:.2}]:", step + 1, step_loss, rolling_avg_loss, grad_norm);
                let test_prompts = [
                    "Даня: Прічс, третя ночі, я втомився. Як стан систем?\nП.Р.І.Ч.С.:",
                    "Даня: Чому виникає перегрів VRM на RTX 3060?\nП.Р.І.Ч.С.:",
                ];
                for p in test_prompts {
                    let prompt_ids: Vec<u16> = tok.encode(p).into_iter().map(|id| id as u16).collect();
                    match model.generate_advanced(&prompt_ids, 120, 0.2, 0.85, 1.03, sft_dataset.header.eos_token_id) {
                        Ok(gen_ids) => {
                            let gen_u32: Vec<u32> = gen_ids.into_iter().map(|id| id as u32).collect();
                            let text = tok.decode(&gen_u32);
                            println!("   💬 \"{}\"", text.trim());
                        }
                        Err(e) => println!("   ⚠️ Помилка генерації: {}", e),
                    }
                }
                println!();
            }
        }

        // Honest Best Checkpoint Selection via Rolling Avg Loss (50)
        if step >= 50 && rolling_loss_window.len() >= 30 && rolling_avg_loss < best_rolling_loss {
            best_rolling_loss = rolling_avg_loss;
            best_loss = step_loss;
            model.sync_weights_from_engine(&engine)?;
            model.save_checkpoint(Path::new(&best_sft_path), step + 1, rolling_avg_loss)?;
            println!("⭐ [BEST CHECKPOINT] Збережено новий рекорд Rolling Avg Loss (50): {:.4} (Step {}) -> {}", best_rolling_loss, step + 1, best_sft_path);
        }

        // Periodic Checkpoint saving every 200 steps
        if (step + 1) % 200 == 0 || step == max_steps - 1 {
            model.sync_weights_from_engine(&engine)?;
            let ckpt_step_path = format!("checkpoints/{}_sft_step{:04}.prichs", mode, step + 1);
            model.save_checkpoint(Path::new(&ckpt_step_path), step + 1, step_loss)?;
            println!("💾 [CHECKPOINT] Збережено контрольну точку SFT: {} (Step Loss: {:.4}, Roll50: {:.4})", ckpt_step_path, step_loss, rolling_avg_loss);
        }
    }

    let total_time = train_start.elapsed();
    println!("\n🎉 SFT ФАЙНТЮНІНГ УСПІШНО ЗАВЕРШЕНО!");
    println!("   > Загальний час: {:.2?}", total_time);
    println!("   > Найкращий Loss: {:.4}", best_loss);

    model.sync_weights_from_engine(&engine)?;
    let final_ckpt = format!("checkpoints/{}_sft_final.prichs", mode);
    model.save_checkpoint(Path::new(&final_ckpt), max_steps, best_loss)?;
    println!("💾 [FINAL] Збережено фінальну модель SFT: {}", final_ckpt);

    Ok(())
}
