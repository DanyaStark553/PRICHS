use anyhow::{Context, Result};
use prichs_model::{PrichsConfig, PrichsModel};
use prichs_tokenizer::PrichsTokenizer;
use std::path::Path;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-9)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = vec![0.0f32; logits.len()];
    let mut sum = 0.0f32;
    for (i, &l) in logits.iter().enumerate() {
        let e = (l - max_val).exp();
        probs[i] = e;
        sum += e;
    }
    let inv_sum = 1.0 / (sum + 1e-9);
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }
    probs
}

fn top_k_indices(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

fn main() -> Result<()> {
    println!("========================================================");
    println!("  🔬 П.Р.І.Ч.С. v2 // ДІАГНОСТИКА №8: GPU vs CPU PARITY");
    println!("  Перевірка ідентичності логітів та семплінгу на SFT");
    println!("========================================================\n");

    let ckpt_path = Path::new("checkpoints/v2_sft_best.prichs");
    if !ckpt_path.exists() {
        eprintln!("Помилка: {:?} не знайдено!", ckpt_path);
        std::process::exit(1);
    }

    let tok_path = Path::new("tokenizer.json");
    let tokenizer = PrichsTokenizer::from_huggingface_json(tok_path)
        .context("Не вдалося завантажити tokenizer.json")?;

    let cfg = PrichsConfig::v2_moe();
    let mut model = PrichsModel::new(cfg)?;

    println!("⏳ [1/4] Завантаження ваг з {:?}...", ckpt_path);
    let (step, loss) = model.load_checkpoint(ckpt_path)?;
    println!("   > Чекпоінт: крок {}, збережений Loss: {:.4}", step, loss);

    // 1. Перевірка router_bias кожного шару
    println!("\n🔍 [2/4] АУДИТ ROUTER_BIAS ПО ВСІХ ШАРАХ:");
    let mut all_biases_zero = true;
    for (l, block) in model.blocks.iter().enumerate() {
        let b = &block.moe.router_bias;
        let non_zero = b.iter().any(|&v| v != 0.0);
        if non_zero {
            all_biases_zero = false;
            println!("   > Шар {:02}: router_bias = {:?}", l, b);
        }
    }
    if all_biases_zero {
        println!("   > ВСІ router_bias на всіх 16 шарах дорівнюють [0.0; 8]!");
        println!("   > ВИСНОВОК: Auxiliary-Loss-Free Dynamic Load Balancing у SFT не використовує зміщення (bias = 0), тому розбіжності через bias виключено.");
    }

    // 2. Тест ідентичності логітів: GPU vs CPU
    println!("\n⚡ [3/4] ТЕСТ ІДЕНТИЧНОСТІ ЛОГІТІВ: GPU Forward vs CPU Forward");
    let test_prompt = "Даня: Прічс, третя ночі, я випив уже четвертий подвійний еспресо, а цей баг із гонкою потоків досі живий.\nП.Р.І.Ч.С.: ";
    let prompt_tokens: Vec<u16> = tokenizer.encode(test_prompt).into_iter().map(|id| id as u16).collect();
    let seq_len = prompt_tokens.len();
    let test_tokens = &prompt_tokens[..seq_len];
    println!("   > Повна послідовність токенів (довжина {}): {:?}", seq_len, test_tokens);

    // CPU Forward
    let t_cpu = std::time::Instant::now();
    let cpu_last_logits = model.forward_last_token(test_tokens, None)?;
    let cpu_dur = t_cpu.elapsed();
    println!("   > CPU Forward виконано за {:.2?}", cpu_dur);

    // GPU Forward
    let t_gpu_init = std::time::Instant::now();
    let mut engine = model.create_gpu_engine()?;
    println!("   > GPU Persistent Engine алоковано за {:.2?}", t_gpu_init.elapsed());

    let t_gpu_fwd = std::time::Instant::now();
    let gpu_all_logits = model.forward_gpu_only(&mut engine, test_tokens)?;
    let gpu_dur = t_gpu_fwd.elapsed();
    println!("   > GPU Forward виконано за {:.2?}", gpu_dur);

    let vocab_size = model.config.vocab_size;
    let gpu_last_logits = &gpu_all_logits[(seq_len - 1) * vocab_size..seq_len * vocab_size];

    // Порівняння метрик логітів
    let mut max_abs_diff = 0.0f32;
    let mut max_diff_idx = 0usize;
    let mut sum_abs_diff = 0.0f32;

    for i in 0..vocab_size {
        let diff = (gpu_last_logits[i] - cpu_last_logits[i]).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
            max_diff_idx = i;
        }
        sum_abs_diff += diff;
    }
    let mean_abs_diff = sum_abs_diff / (vocab_size as f32);
    let cos_sim = cosine_similarity(gpu_last_logits, &cpu_last_logits);

    let cpu_probs = softmax(&cpu_last_logits);
    let gpu_probs = softmax(gpu_last_logits);

    let cpu_top5 = top_k_indices(&cpu_last_logits, 5);
    let gpu_top5 = top_k_indices(gpu_last_logits, 5);

    println!("\n📊 [РЕЗУЛЬТАТИ ПОРІВНЯННЯ GPU vs CPU]:");
    println!("   > Cosine Similarity: {:.6} (1.000000 = ідеальний збіг)", cos_sim);
    println!("   > Max Absolute Diff: {:.6} (токен ID {})", max_abs_diff, max_diff_idx);
    println!("   > Mean Absolute Diff: {:.6}", mean_abs_diff);

    println!("\n   🏆 TOP-5 ТОКЕНІВ НА CPU:");
    for (rank, &(id, logit)) in cpu_top5.iter().enumerate() {
        let tok_str = tokenizer.decode(&[id as u32]);
        println!("      {}. ID {:5} ({:20}) | Logit: {:8.3} | Prob: {:6.2}%", rank + 1, id, repr_str(&tok_str), logit, cpu_probs[id] * 100.0);
    }

    println!("\n   🏆 TOP-5 ТОКЕНІВ НА GPU:");
    for (rank, &(id, logit)) in gpu_top5.iter().enumerate() {
        let tok_str = tokenizer.decode(&[id as u32]);
        println!("      {}. ID {:5} ({:20}) | Logit: {:8.3} | Prob: {:6.2}%", rank + 1, id, repr_str(&tok_str), logit, gpu_probs[id] * 100.0);
    }

    if cpu_top5[0].0 == gpu_top5[0].0 {
        println!("\n   ✅ GREEDY ТОКЕН ЗБІГАЄТЬСЯ НА 100%! (ID: {}, '{}')", cpu_top5[0].0, repr_str(&tokenizer.decode(&[cpu_top5[0].0 as u32])));
    } else {
        println!("\n   ⚠️ GREEDY ТОКЕН ВІДРІЗНЯЄТЬСЯ! CPU ID {} vs GPU ID {}", cpu_top5[0].0, gpu_top5[0].0);
    }

    // 3. Тест семплінгу на 5 промптах (In-Distribution та Out-of-Distribution)
    println!("\n========================================================");
    println!("🎭 [4/4] ТЕСТ СЕМПЛІНГУ: GREEDY vs LOW-TEMP vs HIGH-TEMP");
    println!("========================================================\n");

    let test_prompts = [
        (
            "IN-DISTRIB 1 (Зразок 0 з SFT датасету)",
            "Даня: Прічс, третя ночі, я випив уже четвертий подвійний еспресо, а цей баг із гонкою потоків досі живий. Завари мені ще один, або я почну розмовляти з осцилографом.\nП.Р.І.Ч.С.: "
        ),
        (
            "IN-DISTRIB 2 (Зразок 1 з SFT датасету)",
            "Даня: Паяльна станція виставлена на триста вісімдесят градусів, а припій на земляному полігоні лежить шматком мертвого пластиліну. Я зараз відірву цю доріжку к бісовій матері!\nП.Р.І.Ч.С.: "
        ),
        (
            "OUT-OF-DISTRIB 1 (Привітання та стан)",
            "Даня: Привіт! Як стан систем?\nП.Р.І.Ч.С.: "
        ),
        (
            "OUT-OF-DISTRIB 2 (Кіно / Тарантіно)",
            "Даня: Який твій улюблений фільм Тарантіно?\nП.Р.І.Ч.С.: "
        ),
        (
            "OUT-OF-DISTRIB 3 (Архітектура StarkOS)",
            "Даня: Як влаштована архітектура ядра StarkOS?\nП.Р.І.Ч.С.: "
        ),
    ];

    let eos_id = 2u16;

    for (label, p) in test_prompts {
        println!("------------------------------------------------------------");
        println!("📌 [{}]", label);
        println!("ПРОМПТ: \"{}\"", p.replace('\n', " "));
        let p_ids: Vec<u16> = tokenizer.encode(p).into_iter().map(|id| id as u16).collect();

        // 1. Greedy (temp = 0.0)
        let gen_greedy = model.generate_advanced(&p_ids, 80, 0.0, 1.0, 1.0, eos_id)?;
        let gen_greedy_u32: Vec<u32> = gen_greedy.into_iter().map(|id| id as u32).collect();
        let text_greedy = tokenizer.decode(&gen_greedy_u32);
        let resp_greedy = extract_response(p, &text_greedy);
        println!("\n  🔹 [GREEDY // temp = 0.0]:\n     \"{}\"", resp_greedy.replace('\n', "\n     "));

        // 2. Low-Temp (temp = 0.2, top_p = 0.9, rep_penalty = 1.05)
        let gen_low = model.generate_advanced(&p_ids, 80, 0.2, 0.9, 1.05, eos_id)?;
        let gen_low_u32: Vec<u32> = gen_low.into_iter().map(|id| id as u32).collect();
        let text_low = tokenizer.decode(&gen_low_u32);
        let resp_low = extract_response(p, &text_low);
        println!("\n  🔹 [LOW-TEMP // temp = 0.2, top_p = 0.9, rep = 1.05]:\n     \"{}\"", resp_low.replace('\n', "\n     "));

        // 3. High-Temp (вчорашній режим: temp = 0.7, top_p = 0.9, rep_penalty = 1.15)
        let gen_high = model.generate_advanced(&p_ids, 80, 0.7, 0.9, 1.15, eos_id)?;
        let gen_high_u32: Vec<u32> = gen_high.into_iter().map(|id| id as u32).collect();
        let text_high = tokenizer.decode(&gen_high_u32);
        let resp_high = extract_response(p, &text_high);
        println!("\n  🔹 [HIGH-TEMP // temp = 0.7, top_p = 0.9, rep = 1.15]:\n     \"{}\"", resp_high.replace('\n', "\n     "));
    }

    println!("\n========================================================");
    println!("✅ ДІАГНОСТИКУ ЗАВЕРШЕНО!");
    println!("========================================================");

    Ok(())
}

fn repr_str(s: &str) -> String {
    s.replace('\n', "\\n").replace('\r', "\\r")
}

fn extract_response<'a>(prompt: &str, full: &'a str) -> &'a str {
    if full.starts_with(prompt) {
        &full[prompt.len()..]
    } else {
        full
    }
}
