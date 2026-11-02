use anyhow::Result;
use prichs_cuda::GpuModelEngine;
use prichs_model::{PrichsConfig, PrichsModel};
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
    
    let attn_params = cfg.hidden_dim * cfg.hidden_dim // q
        + cfg.hidden_dim * (cfg.num_kv_heads * (cfg.hidden_dim / cfg.num_attention_heads)) // k
        + cfg.hidden_dim * (cfg.num_kv_heads * (cfg.hidden_dim / cfg.num_attention_heads)) // v
        + cfg.hidden_dim * cfg.hidden_dim; // out
    let norms_params = cfg.hidden_dim * 2; // input_norm + post_attn_norm
    let router_params = cfg.hidden_dim * cfg.num_experts;
    let expert_params = 3 * cfg.hidden_dim * cfg.intermediate_dim;
    let active_moe_params = router_params 
        + expert_params * cfg.num_shared_experts 
        + expert_params * cfg.num_experts_per_tok;
        
    let active_per_layer = attn_params + norms_params + active_moe_params;
    let active = wte_params + cfg.num_layers * active_per_layer + final_norm_params;
    
    (total, active)
}

fn run_vram_benchmark(cfg: PrichsConfig, name: &str, steps: usize, seq_len: usize) -> Result<()> {
    println!("\n========================================================");
    println!("  🚀 PERSISTENT GPU VRAM ENGINE БЕНЧМАРК: {}", name);
    println!("  Шарів: {}, d_model: {}, d_ffn: {}, Експертів: {} (Top-{})", 
        cfg.num_layers, cfg.hidden_dim, cfg.intermediate_dim, cfg.num_experts, cfg.num_experts_per_tok);
    println!("  Послідовність (seq_len): {} токенів", seq_len);
    println!("========================================================");

    let (vram_free_start, vram_total) = GpuModelEngine::get_gpu_memory_info().unwrap_or((0, 0));
    let (ram_init, _commit_init) = get_process_memory_mb();
    println!("📊 Початковий стан: RAM процесу = {:.1} MB, Вільно у VRAM = {:.2} GB / {:.2} GB",
        ram_init, vram_free_start as f64 / (1024.0 * 1024.0 * 1024.0), vram_total as f64 / (1024.0 * 1024.0 * 1024.0));

    let t0 = Instant::now();
    println!("\n⏳ [1/3] Створення моделі на хості...");
    let mut model = PrichsModel::new(cfg.clone())?;
    let (total_params, active_params) = count_parameters(&model);
    println!("   > Загальних параметрів: {:.2}M ({})", total_params as f64 / 1e6, total_params);
    println!("   > Активних параметрів на токен: {:.2}M ({})", active_params as f64 / 1e6, active_params);
    println!("   > Мастер-ваги моделі: {:.2} MB", (total_params * 4) as f64 / (1024.0 * 1024.0));
    println!("   > Час створення на хості: {:.2?}", t0.elapsed());

    let (ram_model, _) = get_process_memory_mb();
    println!("   > Системна RAM процесу: {:.2} MB ({:.2} GB)", ram_model, ram_model / 1024.0);

    println!("\n⏳ [2/3] Алокація та ініціалізація Persistent GPU VRAM Engine...");
    let t_engine = Instant::now();
    let mut engine = model.create_gpu_engine()?;
    let (vram_free_after_alloc, _) = GpuModelEngine::get_gpu_memory_info().unwrap_or((0, 0));
    let vram_used_by_model = (vram_free_start.saturating_sub(vram_free_after_alloc)) as f64 / (1024.0 * 1024.0);
    println!("   > Час ініціалізації VRAM Engine: {:.2?}", t_engine.elapsed());
    println!("   > Фактично зайнято у VRAM моделі (W + dW + m + v + workspace): {:.2} MB ({:.2} GB)",
        vram_used_by_model, vram_used_by_model / 1024.0);
    println!("   > Залишок вільної VRAM на RTX 3060: {:.2} GB",
        vram_free_after_alloc as f64 / (1024.0 * 1024.0 * 1024.0));

    // Synthetic training batch
    let inputs: Vec<u16> = (0..seq_len).map(|i| ((i * 37 + 13) % cfg.vocab_size) as u16).collect();
    let targets: Vec<u16> = (0..seq_len).map(|i| ((i * 37 + 14) % cfg.vocab_size) as u16).collect();
    let mask = vec![1.0f32; seq_len];

    println!("\n🚀 [3/3] Запуск {} тренувальних кроків прямо у VRAM (Zero PCIe Weight Copy)...", steps);
    
    // Warmup step 1
    println!("   > Крок 1 (Warmup)...");
    let step_start = Instant::now();
    let loss1 = model.train_step_vram(&mut engine, &inputs, &targets, &mask, 0.0001, 1)?;
    let step1_time = step_start.elapsed();
    let (ram_step1, _) = get_process_memory_mb();
    let tok_s1 = seq_len as f64 / step1_time.as_secs_f64();
    println!("     Loss: {:.4} | Час кроку: {:.2?} | Швидкість: {:.1} tok/s | RAM: {:.1} MB",
        loss1, step1_time, tok_s1, ram_step1);

    let mut total_time = std::time::Duration::ZERO;
    let mut losses = Vec::with_capacity(steps);
    losses.push(loss1);

    for s in 2..=steps {
        let s_start = Instant::now();
        let loss = model.train_step_vram(&mut engine, &inputs, &targets, &mask, 0.0001, s)?;
        let step_dur = s_start.elapsed();
        total_time += step_dur;
        losses.push(loss);
        
        let (ram_now, _) = get_process_memory_mb();
        let (vram_free_now, _) = GpuModelEngine::get_gpu_memory_info().unwrap_or((0, 0));
        let tok_s = seq_len as f64 / step_dur.as_secs_f64();
        println!("   > Крок {:02}/{} | Loss: {:.4} | Час: {:.2?} | Швидкість: {:.1} tok/s | RAM: {:.1} MB | VRAM Free: {:.2} GB",
            s, steps, loss, step_dur, tok_s, ram_now, vram_free_now as f64 / (1024.0 * 1024.0 * 1024.0));
    }

    let count = (steps.saturating_sub(1)).max(1) as u32;
    let avg_dur = if steps > 1 { total_time / count } else { step_start.elapsed() };
    let avg_tok_s = seq_len as f64 / avg_dur.as_secs_f64();
    let (ram_final, _) = get_process_memory_mb();

    println!("\n🏆 ПІДСУМОК ДЛЯ ПРОФІЛЮ {}:", name);
    println!("   > Середній час кроку:       {:.2?}", avg_dur);
    println!("   > Реальна швидкість:        {:.1} tok/s", avg_tok_s);
    println!("   > Системна RAM процесу:     {:.2} MB ({:.2} GB)  <-- РОЗВАНТАЖЕНО!", ram_final, ram_final / 1024.0);
    println!("   > Динаміка Loss:            {:.4} -> {:.4} -> {:.4}", losses[0], losses[1], losses.last().unwrap());
    
    // Time estimates
    let sec_10m = 10_000_000.0 / avg_tok_s;
    let sec_20m = 20_000_000.0 / avg_tok_s;
    let sec_50m = 50_000_000.0 / avg_tok_s;
    println!("   > Прогноз для 10M токенів:  {:.1} годин ({:.1} діб)", sec_10m / 3600.0, sec_10m / 86400.0);
    println!("   > Прогноз для 20M токенів:  {:.1} годин ({:.1} діб)", sec_20m / 3600.0, sec_20m / 86400.0);
    println!("   > Прогноз для 50M токенів:  {:.1} годин ({:.1} діб)", sec_50m / 3600.0, sec_50m / 86400.0);

    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("v2");
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    match mode {
        "baby" => {
            let cfg = PrichsConfig::baby();
            run_vram_benchmark(cfg, "Baby-П.Р.І.Ч.С. (52M Total / 19M Active)", steps, 1024)?;
        }
        "v2" | "472m" => {
            let cfg = PrichsConfig::v2_moe();
            run_vram_benchmark(cfg, "П.Р.І.Ч.С. v2 Heavy MoE (472M Total / 160M Active)", steps, 1024)?;
        }
        "mid" | "355m" | "362m" => {
            let mut cfg = PrichsConfig::v2_moe();
            cfg.num_layers = 12; // 12 layers instead of 16 (~355M Total / 121M Active)
            run_vram_benchmark(cfg, "П.Р.І.Ч.С. v2 Balanced (12 layers / 355M Total)", steps, 1024)?;
        }
        other => {
            println!("Невідомий режим: {}. Доступні: 'baby', 'v2', 'mid'", other);
        }
    }

    Ok(())
}
