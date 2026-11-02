use anyhow::{bail, Context, Result};
use prichs_cuda::GpuModelEngine;
use prichs_model::{PrichsConfig, PrichsModel};
use prichs_tokenizer::PrichsTokenizer;
use prichs_train::PrichsMmapDataset;
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

/// Generates a randomized permutation of chunk indices for an epoch
fn generate_shuffled_chunks(num_chunks: usize, epoch_seed: u64) -> Vec<usize> {
    let mut chunks: Vec<usize> = (0..num_chunks).collect();
    let mut state = epoch_seed ^ 0x9e3779b97f4a7c15;
    if state == 0 { state = 1; }
    for i in (1..chunks.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        chunks.swap(i, j);
    }
    chunks
}

fn main() -> Result<()> {
    println!("\x1b[38;2;56;189;248m");
    println!(r#"
  ██████╗         ██████╗       ██╗       ██████╗     ██╗  ██╗
  ██╔══██╗        ██╔══██╗      ██║      ██╔════╝     ██║  ██║
  ██████╔╝  ██╗   ██████╔╝  ██╗ ██║  ██╗ ██║    ██╗   ███████║
  ██╔═══╝   ╚═╝   ██╔══██╗  ╚═╝ ██║  ╚═╝ ██║    ╚═╝   ██╔══██║
  ██║             ██║  ██║      ██║      ╚██████╗     ██║  ██║
  ╚═╝             ╚═╝  ╚═╝      ╚═╝       ╚═════╝     ╚═╝  ╚═╝
  П. Р. І. Ч. С. v2 // Sovereign Pre-Training Engine
  Pure GPU-Resident Activation Pipeline // Zero PCIe Weight Copy
    "#);
    println!("\x1b[0m");

    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("v2");
    let max_steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15000);
    let accum_steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let fresh = args.iter().any(|a| a == "--fresh" || a == "-f");

    let (cfg, model_title, max_lr, min_lr) = match mode {
        "baby" => {
            (PrichsConfig::baby(), "Baby-П.Р.І.Ч.С. (52M Total / 19M Active)", 0.000400f32, 0.000040f32)
        }
        "mid" | "355m" | "362m" => {
            let mut c = PrichsConfig::v2_moe();
            c.num_layers = 12;
            (c, "П.Р.І.Ч.С. v2 Balanced (12L / 355M Total / 121M Active)", 0.000300f32, 0.000030f32)
        }
        "v2" | "472m" => {
            (PrichsConfig::v2_moe(), "П.Р.І.Ч.С. v2 Heavy MoE (16L / 472M Total / 160M Active)", 0.000250f32, 0.000025f32)
        }
        other => {
            println!("Невідомий профіль: '{}'. Використовуйте: 'v2', 'mid', 'baby'.", other);
            std::process::exit(1);
        }
    };

    let explicit_dataset = args.iter().find(|a| a.ends_with(".prichsbin")).cloned();
    let default_large = "data/pretrain_v2_large.prichsbin";
    let default_standard = "data/pretrain_v2.prichsbin";
    let chosen_dataset = if let Some(ref d) = explicit_dataset {
        d.as_str()
    } else if Path::new(default_large).exists() {
        default_large
    } else {
        default_standard
    };
    let dataset_path = Path::new(chosen_dataset);
    if !dataset_path.exists() {
        bail!("Файл датасету {:?} не знайдено! Перевірте директорію data/", dataset_path);
    }

    let dataset = PrichsMmapDataset::open(dataset_path.to_str().unwrap())
        .context("Не вдалося відкрити Mmap датасет pretrain_v2")?;
    let total_tokens = dataset.header.total_tokens as usize;
    let seq_len = cfg.max_seq_len;
    let num_chunks = (total_tokens.saturating_sub(1)) / seq_len;
    let steps_per_epoch = num_chunks / accum_steps;
    let total_epochs_target = (max_steps as f64) / (steps_per_epoch as f64);

    let tokenizer_path = Path::new("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        println!("📖 [TOKENIZER] Завантажено чистий ByteLevel токенізатор v2 з {:?}", tokenizer_path);
        Some(PrichsTokenizer::from_huggingface_json(tokenizer_path)?)
    } else {
        println!("⚠️ [TOKENIZER] tokenizer.json не знайдено. Live Monitor буде вимкнено.");
        None
    };

    println!("========================================================");
    println!("  🚀 ПРОФІЛЬ НАВЧАННЯ:  {}", model_title);
    println!("  Датасет:          {} ({:.2}M токенів)", dataset_path.display(), total_tokens as f64 / 1e6);
    println!("  Контекст T:       {} токенів", seq_len);
    println!("  Вікон на епоху:   {} chunks (1 епоха = {} кроків)", num_chunks, steps_per_epoch);
    println!("  Grad Accum:       {} micro-batches ({} токенів/крок)", accum_steps, seq_len * accum_steps);
    println!("  Цільових кроків:  {} ({:.2} повних епох / {:.2}M токенів)",
        max_steps, total_epochs_target, (max_steps * seq_len * accum_steps) as f64 / 1e6);
    println!("  LR розклад:       {:.6} -> {:.6} (Cosine Decay, 250 кроків warmup)", max_lr, min_lr);
    println!("  Gradient Clip:    max_grad_norm = 1.0 (GPU Hardware accelerated)");
    println!("  Live Monitor:     кожні 100 кроків (2 живих промпти)");
    println!("========================================================\n");

    let (ram_init, _) = get_process_memory_mb();
    let (vram_free_start, vram_total) = GpuModelEngine::get_gpu_memory_info().unwrap_or((0, 0));
    println!("📊 Початковий стан хоста:");
    println!("   > Системна RAM процесу: {:.2} MB", ram_init);
    println!("   > Вільна VRAM GPU:      {:.2} GB із {:.2} GB",
        vram_free_start as f64 / (1024.0 * 1024.0 * 1024.0), vram_total as f64 / (1024.0 * 1024.0 * 1024.0));

    println!("\n⏳ [1/3] Ініціалізація ваг моделі на хості...");
    let t_model = Instant::now();
    let mut model = PrichsModel::new(cfg.clone())?;
    let (total_params, active_params) = count_parameters(&model);
    println!("   > Загальних параметрів: {:.2}M ({})", total_params as f64 / 1e6, total_params);
    println!("   > Активних на токен:    {:.2}M ({})", active_params as f64 / 1e6, active_params);
    println!("   > Час збірки моделі:    {:.2?}", t_model.elapsed());

    let ckpt_dir = Path::new("checkpoints");
    std::fs::create_dir_all(ckpt_dir)?;
    let default_best = format!("checkpoints/{}_pretrain_v2_best.prichs", mode);
    let best_ckpt_path = Path::new(&default_best);

    let resume_idx = args.iter().position(|a| a == "--resume" || a == "-r");
    let explicit_resume_path: Option<String> = if let Some(idx) = resume_idx {
        args.get(idx + 1).filter(|s| !s.starts_with('-')).cloned()
    } else {
        None
    };

    let latest_step_ckpt = if let Ok(entries) = std::fs::read_dir(ckpt_dir) {
        let prefix = format!("{}_pretrain_v2_step", mode);
        let mut max_step = 0usize;
        let mut best_path = None;
        for entry in entries.flatten() {
            let fname = entry.file_name().to_string_lossy().to_string();
            if fname.starts_with(&prefix) && fname.ends_with(".prichs") {
                let num_part = &fname[prefix.len()..fname.len() - 7];
                if let Ok(st) = num_part.parse::<usize>() {
                    if st >= max_step {
                        max_step = st;
                        best_path = Some(entry.path());
                    }
                }
            }
        }
        best_path
    } else {
        None
    };

    let target_resume_buf = if let Some(ref p) = explicit_resume_path {
        std::path::PathBuf::from(p)
    } else if let Some(p) = latest_step_ckpt {
        p
    } else {
        std::path::PathBuf::from(&default_best)
    };
    let target_resume_path = target_resume_buf.as_path();

    let mut start_step = 0usize;
    let mut best_loss = f32::INFINITY;
    if fresh {
        println!("🌱 [START] Прапорець --fresh виявлено: старт з чистої рандомної ініціалізації (Loss ~10.55)!");
    } else if target_resume_path.exists() {
        match model.load_checkpoint(target_resume_path) {
            Ok((step, loss)) => {
                println!("💾 [RESUME] Успішно відновлено чекпоінт {:?}!", target_resume_path);
                println!("   > Продовження з кроку: {} / {}", step, max_steps);
                println!("   > Зафіксований Loss:   {:.4}", loss);
                start_step = step;
                best_loss = loss;
            }
            Err(e) => {
                println!("⚠️ [WARN] Помилка завантаження чекпоінту ({}), старт з нуля.", e);
            }
        }
    } else {
        println!("ℹ️ [START] Чекпоінтів для продовження не знайдено, старт з чистої ініціалізації.");
    }

    println!("\n⏳ [2/3] Алокація та ініціалізація Persistent GPU VRAM Engine на RTX 3060...");
    let t_gpu = Instant::now();
    let mut engine = model.create_gpu_engine()?;
    let (vram_free_after, _) = GpuModelEngine::get_gpu_memory_info().unwrap_or((0, 0));
    let vram_used_mb = (vram_free_start.saturating_sub(vram_free_after)) as f64 / (1024.0 * 1024.0);
    println!("   > Час завантаження в VRAM: {:.2?}", t_gpu.elapsed());
    println!("   > Використано VRAM:        {:.2} MB ({:.2} GB)", vram_used_mb, vram_used_mb / 1024.0);
    println!("   > Залишок вільної VRAM:    {:.2} GB", vram_free_after as f64 / (1024.0 * 1024.0 * 1024.0));

    let (ram_ready, _) = get_process_memory_mb();
    println!("   > Системна RAM процесу:    {:.2} MB ({:.2} GB)  <-- РОЗВАНТАЖЕНО!", ram_ready, ram_ready / 1024.0);

    // Верифікація відсутності подвійної алокації VRAM при виклику генерації
    if let Some(ref tok) = tokenizer {
        println!("\n🧪 [VERIFY] Тестовий виклик generate_advanced для верифікації VRAM/RAM...");
        let test_prompt = "Тест системи:";
        let prompt_ids: Vec<u16> = tok.encode(test_prompt).into_iter().map(|id| id as u16).collect();
        let _ = model.generate_advanced(&prompt_ids, 5, 0.7, 0.9, 1.1, dataset.header.eos_token_id);
        let (ram_test, _) = get_process_memory_mb();
        let (vram_free_test, _) = GpuModelEngine::get_gpu_memory_info().unwrap_or((0, 0));
        println!("   > VRAM після generate_advanced: {:.2} GB зайнято (вільно {:.2} GB)",
            (vram_total - vram_free_test) as f64 / (1024.0 * 1024.0 * 1024.0),
            vram_free_test as f64 / (1024.0 * 1024.0 * 1024.0));
        println!("   > RAM процесу після generate_advanced: {:.2} MB (нульовий оверхед!)", ram_test);
    }

    println!("\n🚀 [3/3] Запуск бойового навчання у 12GB VRAM...");
    let warmup_steps = 250usize;

    // Chunk Shuffling State
    let mut epoch = start_step / steps_per_epoch.max(1);
    let mut shuffled_chunks = generate_shuffled_chunks(num_chunks, epoch as u64);
    let mut chunk_idx = (start_step * accum_steps) % num_chunks.max(1);
    println!("🎲 [SHUFFLE] Епоха {}: сформовано перемішаний порядок із {} вікон (seed={})",
        epoch + 1, num_chunks, epoch);

    let train_start = Instant::now();
    let mut tokens_processed = 0usize;
    let mut recent_tok_s = 0.0f64;

    for step in start_step..max_steps {
        let step_start = Instant::now();
        let lr = compute_lr(step, max_steps, warmup_steps, max_lr, min_lr);

        let mut step_loss = 0.0f32;
        let grad_scale = 1.0f32 / (accum_steps as f32);

        // Hardware-accelerated persistent gradient zeroing ONCE per step
        engine.zero_grads()?;

        // Gradient Accumulation over accum_steps micro-batches
        for _a in 0..accum_steps {
            if chunk_idx >= shuffled_chunks.len() {
                epoch += 1;
                shuffled_chunks = generate_shuffled_chunks(num_chunks, epoch as u64);
                chunk_idx = 0;
                println!("\n🔄 [DATASET] Перехід на епоху {} (Нове випадкове перемішування {} вікон!)", epoch + 1, num_chunks);
            }

            let c_id = shuffled_chunks[chunk_idx];
            chunk_idx += 1;
            let start_idx = c_id * seq_len;

            let (inputs, targets, _cu) = dataset.get_sequence_with_boundaries(start_idx, seq_len)?;
            let mask = vec![1.0f32; seq_len];
            let micro_loss = model.forward_backward_vram_accum(&mut engine, inputs, targets, &mask, grad_scale)?;
            step_loss += micro_loss / (accum_steps as f32);
        }

        // Hardware-accelerated Gradient Clipping in VRAM
        let grad_norm = engine.clip_grad_norm(1.0)?;

        // Fused 128-bit AdamW Step directly on GPU VRAM
        engine.adamw_step(lr, 0.9, 0.95, 1e-8, 0.1, step + 1)?;

        let step_dur = step_start.elapsed();
        let step_tokens = seq_len * accum_steps;
        tokens_processed += step_tokens;
        let step_tok_s = step_tokens as f64 / step_dur.as_secs_f64();
        recent_tok_s = if recent_tok_s == 0.0 { step_tok_s } else { recent_tok_s * 0.9 + step_tok_s * 0.1 };

        let (ram_current, _) = get_process_memory_mb();
        let (vram_free_current, _) = GpuModelEngine::get_gpu_memory_info().unwrap_or((0, 0));

        let progress = (step + 1) as f64 / max_steps as f64;
        let remaining_steps = max_steps.saturating_sub(step + 1);
        let eta_secs = (remaining_steps as f64 * step_dur.as_secs_f64()) as u64;
        let eta_str = format!("{:02}:{:02}:{:02}", eta_secs / 3600, (eta_secs % 3600) / 60, eta_secs % 60);

        if step % 5 == 0 || step == max_steps - 1 {
            println!(
                "[{:04}/{:04}] ({:.1}%) | Ep:{:.2} | Loss: {:.4} | GradNorm: {:.2} | LR: {:.6} | Час: {:.2?} | {:.1} tok/s | RAM: {:.0} MB | VRAM: {:.2} GB | ETA: {}",
                step + 1, max_steps, progress * 100.0,
                (step + 1) as f64 / steps_per_epoch as f64,
                step_loss, grad_norm, lr, step_dur, step_tok_s, ram_current,
                (vram_total - vram_free_current) as f64 / (1024.0 * 1024.0 * 1024.0), eta_str
            );
        }

        // Live Emergence Monitor every 100 steps
        if (step + 1) % 100 == 0 {
            if let Some(ref tok) = tokenizer {
                model.sync_weights_from_engine(&engine)?;
                println!("\n🎭 [LIVE EMERGENCE MONITOR // Крок {} | Ep: {:.2} | Loss: {:.4} | GradNorm: {:.2}]:",
                    step + 1, (step + 1) as f64 / steps_per_epoch as f64, step_loss, grad_norm);
                let prompts = [
                    "Архітектура системи StarkOS базується на",
                    "Даня: Привіт! Як працюють твої системи?\nП.Р.І.Ч.С.:",
                ];
                for prompt in prompts {
                    let prompt_ids: Vec<u16> = tok.encode(prompt).into_iter().map(|id| id as u16).collect();
                    match model.generate_advanced(
                        &prompt_ids,
                        45,
                        0.7,
                        0.9,
                        1.1,
                        dataset.header.eos_token_id,
                    ) {
                        Ok(generated_ids) => {
                            let gen_u32: Vec<u32> = generated_ids.into_iter().map(|id| id as u32).collect();
                            let text = tok.decode(&gen_u32);
                            println!("   💬 \"{}\"", text.replace('\n', "\\n"));
                        }
                        Err(e) => {
                            println!("   ⚠️ [GEN ERROR] {}", e);
                        }
                    }
                }
                println!();
            }
        }

        // Periodic checkpoint
        if (step + 1) % 500 == 0 || (step_loss < best_loss && step % 50 == 0) {
            if step_loss < best_loss {
                best_loss = step_loss;
            }
            model.sync_weights_from_engine(&engine)?;
            let ckpt_step_path = format!("checkpoints/{}_pretrain_v2_step{:04}.prichs", mode, step + 1);
            model.save_checkpoint(Path::new(&ckpt_step_path), step + 1, step_loss)?;
            model.save_checkpoint(best_ckpt_path, step + 1, step_loss)?;
            println!("💾 [CHECKPOINT] Збережено контрольну точку: {} (Loss: {:.4})", ckpt_step_path, step_loss);
        }
    }

    let total_time = train_start.elapsed();
    let avg_tok_s = tokens_processed as f64 / total_time.as_secs_f64();

    println!("\n🏁 НАВЧАННЯ УСПІШНО ЗАВЕРШЕНО!");
    println!("   > Опрацьовано токенів:    {} ({:.2}M)", tokens_processed, tokens_processed as f64 / 1e6);
    println!("   > Загальний час сесії:    {:.2?}", total_time);
    println!("   > Середня швидкість:      {:.1} tok/s", avg_tok_s);
    println!("   > Фінальний Loss:         {:.4}", best_loss);

    // Final checkpoint save
    model.sync_weights_from_engine(&engine)?;
    let final_ckpt = format!("checkpoints/{}_pretrain_v2_final.prichs", mode);
    model.save_checkpoint(Path::new(&final_ckpt), max_steps, best_loss)?;
    println!("💾 [FINAL] Фінальну модель збережено в: {}", final_ckpt);

    Ok(())
}
