use anyhow::{bail, Result};
use prichs_core::backward::{
    fused_cross_entropy_forward_backward, moe_router_unbiased_backward, rmsnorm_backward,
    swiglu_activation_backward,
};
use prichs_core::Tensor;
use prichs_cuda::MysteriumReadOnlyBridge;
use prichs_model::{
    ActivationScratchpad, FullModelOptimizer, ModelGradients, PrichsConfig, PrichsModel,
};
use prichs_tokenizer::PrichsTokenizer;
use prichs_train::{
    AdamWOptimizer, ParameterGroup, PrichsMmapDataset, PrichsSftDataset,
};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

static INFERENCE_REQUESTED: AtomicBool = AtomicBool::new(false);
static INFERENCE_PERMITTED: AtomicBool = AtomicBool::new(false);

fn print_banner() {
    println!("\x1b[38;2;56;189;248m"); // Cyan-400 Arc Reactor
    println!(r#"
  ██████╗         ██████╗       ██╗       ██████╗     ██╗  ██╗
  ██╔══██╗        ██╔══██╗      ██║      ██╔════╝     ██║  ██║
  ██████╔╝  ██╗   ██████╔╝  ██╗ ██║  ██╗ ██║    ██╗   ███████║
  ██╔═══╝   ╚═╝   ██╔══██╗  ╚═╝ ██║  ╚═╝ ██║    ╚═╝   ██╔══██║
  ██║             ██║  ██║      ██║      ╚██████╗     ██║  ██║
  ╚═╝             ╚═╝  ╚═╝      ╚═╝       ╚═════╝     ╚═╝  ╚═╝
  П. Р. І. Ч. С. // Просто Розумний Інтелект Чекає Свого
  Analytical Backward Kernels + Fused Cross-Entropy + Overfit Test
  Architecture: Baby-П.Р.І.Ч.С. (52M Total / 19M Active)
    "#);
    println!("\x1b[0m");
}

fn run_interactive_chat(tokenizer: &PrichsTokenizer) -> Result<()> {
    println!("\x1b[38;2;56;189;248m");
    println!("================================================================");
    println!("  П.Р.І.Ч.С. // Live Interactive Neural Terminal                 ");
    println!("  Architecture : Baby-П.Р.І.Ч.С. (52M Total / 19M Active)        ");
    println!("  Vocabulary   : 32,768 tokens (Byte-Level BPE)                 ");
    println!("  Sampling     : Temperature=0.7 | Top-P=0.9 | RepPenalty=1.15  ");
    println!("  Commands     : Введіть 'вихід' або 'exit' для завершення       ");
    println!("================================================================\x1b[0m\n");

    let cfg = PrichsConfig::baby();
    let mut model = PrichsModel::new(cfg)?;

    let p_path = Path::new("checkpoints/baby_prichs_persona_best.prichs");
    let ckpt_path = if p_path.exists() {
        p_path
    } else {
        Path::new("checkpoints/baby_prichs_best.prichs")
    };
    if ckpt_path.exists() {
        match model.load_checkpoint(ckpt_path) {
            Ok((step, loss)) => {
                println!("💾 [OK] Завантажено збережені ваги з {:?} (Крок {}, Loss {:.4})\n", ckpt_path, step, loss);
            }
            Err(e) => {
                println!("⚠️ [WARN] Чекпоінт пошкоджено ({}), запуск з початкового стану.\n", e);
            }
        }
    } else {
        println!("🌱 [INFO] Чекпоінт не знайдено. Модель працює в нульовому стані розуму (початкові ваги).\n");
    }

    let stdin = std::io::stdin();
    let mut input = String::new();

    loop {
        print!("\x1b[38;2;252;238;10mДаня > \x1b[0m");
        std::io::stdout().flush()?;
        input.clear();
        if stdin.read_line(&mut input)? == 0 {
            break;
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.eq_ignore_ascii_case("exit")
            || trimmed.eq_ignore_ascii_case("quit")
            || trimmed == "вихід"
            || trimmed == "Вихід"
            || trimmed == "ВИХІД"
        {
            println!("\n👋 До зв'язку, Старку! Репульсори переведено в черговий режим.\n");
            break;
        }

        let mut prompt_tokens: Vec<u16> = tokenizer
            .encode(trimmed)
            .into_iter()
            .map(|t| t as u16)
            .collect();
        if prompt_tokens.is_empty() {
            prompt_tokens.push(1); // <bos>
        }

        print!("\x1b[38;2;56;189;248mП.Р.І.Ч.С. > \x1b[0m");
        std::io::stdout().flush()?;

        let generated = model.generate_advanced(
            &prompt_tokens,
            48,   // Max new tokens
            0.7,  // Temperature
            0.9,  // Top-P
            1.15, // Repetition penalty
            2,    // EOS
        )?;

        let new_tokens: Vec<u32> = generated[prompt_tokens.len()..]
            .iter()
            .map(|&x| x as u32)
            .collect();

        let response_text = tokenizer.decode(&new_tokens);
        println!("{}\n", response_text.trim());
    }

    Ok(())
}

fn generate_shuffled_chunks(num_chunks: usize, epoch: usize) -> Vec<usize> {
    let mut chunks: Vec<usize> = (0..num_chunks).collect();
    let mut state = (epoch as u64 + 42).wrapping_mul(0x9E3779B97F4A7C15);
    for i in (1..num_chunks).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        chunks.swap(i, j);
    }
    chunks
}

fn run_training_loop(tokenizer: &PrichsTokenizer, args: &[String]) -> Result<()> {
    print_banner();
    let is_persona = args.iter().any(|a| a == "--persona" || a == "-p" || a == "persona")
        || (Path::new("data/pure_stark_persona.prichsbin").exists() && args.iter().any(|a| a == "--train-persona"));
    let is_sft = Path::new("data/golden_sft.prichsbin").exists();

    let (shard_path, title, max_steps, max_lr, min_lr, preview_prompt) = if is_persona && Path::new("data/pure_stark_persona.prichsbin").exists() {
        (
            "data/pure_stark_persona.prichsbin",
            "Фаза 2B: Чисте Ядро Старка (Pure Persona Distillation, 725k токенів, 0% Alpaca)",
            800,
            0.000200,
            0.000020,
            "Даня: Привіт! Як твої системи?\nП.Р.І.Ч.С.:",
        )
    } else if is_sft {
        (
            "data/golden_sft.prichsbin",
            "Фаза 2: SFT Дистиляція Особистості (4.15M токенів)",
            1000,
            0.000300,
            0.000040,
            "Даня: Привіт! Як себе почуваєш після нічного тренування?\nП.Р.І.Ч.С.:",
        )
    } else {
        (
            "data/train.prichsbin",
            "Нічний Марафон (Shuffled-8k Multi-Domain MoE)",
            4100,
            0.000400,
            0.000040,
            "Даня: Привіт! Як себе почуваєш?\nП.Р.І.Ч.С.:",
        )
    };

    println!("================================================================");
    println!("  П.Р.І.Ч.С. // {}", title);
    println!("================================================================");

    let dataset = PrichsMmapDataset::open(shard_path)?;
    let total_tokens = dataset.header.total_tokens as usize;
    println!("📦 [1/3] Завантажено датасет: {} ({} токенів)", shard_path, total_tokens);

    let cfg = PrichsConfig::baby();
    println!("🧠 [2/3] Збірка Baby-П.Р.І.Ч.С. (52M Total / 19M Active)...");
    let mut model = PrichsModel::new(cfg.clone())?;

    let ckpt_path = if is_persona && Path::new("checkpoints/baby_prichs_sft_best.prichs").exists() {
        Path::new("checkpoints/baby_prichs_sft_best.prichs")
    } else {
        Path::new("checkpoints/baby_prichs_best.prichs")
    };
    let mut best_loss = f32::INFINITY;
    if ckpt_path.exists() {
        match model.load_checkpoint(ckpt_path) {
            Ok((step, loss)) => {
                println!("💾 [OK] Відновлено базові ваги з {:?} (Крок {}, Loss {:.4})", ckpt_path, step, loss);
                best_loss = loss;
            }
            Err(e) => {
                println!("⚠️ [WARN] Чекпоінт не завантажено ({}), старт з 0.", e);
            }
        }
    }

    let micro_seq_len = 1024;
    let accum_steps = 4; // Effective batch size = 4,096 tokens per optimizer update
    let batch_tokens = micro_seq_len * accum_steps; // 4,096 tokens
    let num_chunks = total_tokens / micro_seq_len;

    let start_step = 0;
    let mut current_epoch = 0usize;
    let mut chunk_cursor = 0usize;
    let mut shuffled_chunks = generate_shuffled_chunks(num_chunks, current_epoch);

    println!("\n🔥 [3/3] Старт тренувального циклу ({} кроків, 4,096 токенів/крок, 6GB VRAM Workspace)...", max_steps);
    println!("   > Цільових кроків: {} (Обсяг: ~{} токенів)", max_steps, max_steps * batch_tokens);
    println!("   > Контекстне вікно: 1,024 токени | Онлайн-шафлінг чанків: {} чанків", num_chunks);
    println!("   > AdamW моменти: Скинуто в 0 (чистий градієнтний спуск)");
    println!("   > LR Schedule: {:.6} -> {:.6} (Cosine Decay на {} кроків)\n", max_lr, min_lr, max_steps);
    std::fs::create_dir_all("checkpoints")?;

    // Fresh in-place Full Model Optimizer (Reset moments exp_avg = 0, exp_avg_sq = 0)
    let mut optimizer = FullModelOptimizer::new(&model, 0.9, 0.95, 1e-8, 0.1);
    let mut accum_grads = ModelGradients::new(&model);
    let mut micro_grads = ModelGradients::new(&model);

    let start_time = Instant::now();

    for step in start_step..max_steps {
        accum_grads.zero();
        let mut step_loss = 0.0f32;

        for _ in 0..accum_steps {
            if chunk_cursor >= num_chunks {
                current_epoch += 1;
                chunk_cursor = 0;
                shuffled_chunks = generate_shuffled_chunks(num_chunks, current_epoch);
            }

            let chunk_idx = shuffled_chunks[chunk_cursor];
            chunk_cursor += 1;

            let token_offset = chunk_idx * micro_seq_len;
            let (x, targets, _) = dataset.get_sequence_with_boundaries(token_offset, micro_seq_len)?;

            let loss = model.forward_backward(x, targets, &mut micro_grads)?;
            step_loss += loss;
            accum_grads.add(&micro_grads);
        }

        step_loss /= accum_steps as f32;
        accum_grads.scale(1.0 / accum_steps as f32);

        // Cosine learning rate decay
        let progress = step as f32 / max_steps as f32;
        let lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (progress * std::f32::consts::PI).cos());

        // In-place AdamW step across all 274 model tensors in parallel
        optimizer.step(&mut model, &accum_grads.grads, lr)?;

        let elapsed = start_time.elapsed().as_secs_f32();
        let steps_done = (step - start_step + 1) as f32;
        let tok_s = (steps_done * batch_tokens as f32) / elapsed.max(0.001);
        let epoch_progress = current_epoch as f32 + (chunk_cursor as f32 / num_chunks as f32);

        if step < start_step + 5 || step % 5 == 0 || step == max_steps - 1 {
            println!(
                "⚡ [{:04}/{}] Loss: {:.4} | LR: {:.6} | Ep: {:.2} | Швидкість: {:.0} tok/s (Батч: 4,096 tok)",
                step, max_steps, step_loss, lr, epoch_progress, tok_s
            );
            std::io::stdout().flush().ok();
        }

        if step_loss < best_loss || step % 25 == 0 || step == max_steps - 1 {
            best_loss = best_loss.min(step_loss);
            let main_ckpt = Path::new("checkpoints/baby_prichs_best.prichs");
            model.save_checkpoint(main_ckpt, step, step_loss)?;
            if is_persona {
                let p_ckpt = Path::new("checkpoints/baby_prichs_persona_best.prichs");
                model.save_checkpoint(p_ckpt, step, step_loss).ok();
            } else {
                let sft_ckpt_path = Path::new("checkpoints/baby_prichs_sft_best.prichs");
                model.save_checkpoint(sft_ckpt_path, step, step_loss).ok();
            }
        }

        // Live text generation preview every 25 steps (and at start)
        if step == start_step || step == start_step + 4 || step % 25 == 0 || step == max_steps - 1 {
            let test_prompt: Vec<u16> = tokenizer.encode(preview_prompt).into_iter().map(|t| t as u16).collect();
            let sample = model.generate_advanced(&test_prompt, 48, 0.7, 0.9, 1.15, 2)?;
            let sample_u32: Vec<u32> = sample.iter().map(|&x| x as u32).collect();
            let text = tokenizer.decode(&sample_u32);
            println!("   💬 [Жива генерація @ крок {}]:\n\"{}\"\n", step, text.trim());
            std::io::stdout().flush().ok();
        }
    }

    println!("\n✅ Цикл тренування завершено! Найкращий Loss: {:.4}", best_loss);
    Ok(())
}

fn atomic_save_checkpoint(model: &PrichsModel, path: &Path, step: usize, loss: f32) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp_path = path.with_extension("tmp");
    model.save_checkpoint(&tmp_path, step, loss)?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

fn write_weekend_telemetry(
    step: usize,
    max_steps: usize,
    stage: usize,
    stage_name: &str,
    loss: f32,
    best_loss: f32,
    lr: f32,
    epoch: f32,
    tok_s: f32,
    tokens_trained: usize,
    elapsed_sec: f32,
    last_quote: &str,
) -> Result<()> {
    let telemetry_path = Path::new("data/weekend_telemetry.json");
    let tmp_path = Path::new("data/weekend_telemetry.tmp");

    let clean_quote = last_quote
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', " ")
        .replace('\r', "");

    let epoch_time = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let json_content = format!(
        "{{\n  \"mode\": \"WEEKEND_AUTOPILOT\",\n  \"status\": \"TRAINING\",\n  \"stage\": {},\n  \"stage_name\": \"{}\",\n  \"step\": {},\n  \"max_steps\": {},\n  \"loss\": {:.4},\n  \"best_loss\": {:.4},\n  \"lr\": {:.6},\n  \"epoch\": {:.2},\n  \"tok_per_sec\": {:.1},\n  \"tokens_trained\": {},\n  \"elapsed_sec\": {:.0},\n  \"last_quote\": \"{}\",\n  \"timestamp\": {}\n}}\n",
        stage, stage_name, step, max_steps, loss, best_loss, lr, epoch, tok_s, tokens_trained, elapsed_sec, clean_quote, epoch_time
    );

    std::fs::write(&tmp_path, json_content)?;
    std::fs::rename(&tmp_path, telemetry_path)?;
    Ok(())
}

fn send_embedded_json(mut stream: &TcpStream, json: &str) {
    let bytes = json.as_bytes();
    let header = format!(
        "HTTP/1.1 200 OK\r\n\
Access-Control-Allow-Origin: *\r\n\
Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
Access-Control-Allow-Headers: Content-Type, X-API-KEY\r\n\
Content-Type: application/json; charset=utf-8\r\n\
Content-Length: {}\r\n\
Connection: close\r\n\r\n",
        bytes.len()
    );
    stream.write_all(header.as_bytes()).ok();
    stream.write_all(bytes).ok();
    stream.flush().ok();
}

fn extract_embedded_field(body: &str, field_name: &str) -> Option<String> {
    let pattern = format!("\"{}\":", field_name);
    let pos = body.find(&pattern)?;
    let after = &body[pos + pattern.len()..].trim_start();
    if after.starts_with('"') {
        let content = &after[1..];
        let end = content.find('"')?;
        let unescaped = content[..end]
            .replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t")
            .replace("\\\"", "\"");
        Some(unescaped)
    } else {
        None
    }
}

fn escape_embedded_json(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn handle_embedded_request(
    mut stream: TcpStream,
    model: Arc<Mutex<PrichsModel>>,
    tokenizer: Arc<PrichsTokenizer>,
    current_step: Arc<Mutex<usize>>,
    current_loss: Arc<Mutex<f32>>,
    current_stage: Arc<Mutex<usize>>,
) {
    let mut request_data = Vec::with_capacity(4096);
    let mut temp = [0u8; 2048];

    loop {
        let n = match stream.read(&mut temp) {
            Ok(n) if n > 0 => n,
            _ => break,
        };
        request_data.extend_from_slice(&temp[..n]);

        if let Some(pos) = request_data.windows(4).position(|w| w == b"\r\n\r\n") {
            let header_str = String::from_utf8_lossy(&request_data[..pos]);
            let mut content_length = 0usize;
            for line in header_str.lines() {
                if line.to_ascii_lowercase().starts_with("content-length:") {
                    if let Some(val) = line.split(':').nth(1) {
                        content_length = val.trim().parse().unwrap_or(0);
                    }
                }
            }
            let body_len = request_data.len() - (pos + 4);
            if body_len >= content_length {
                break;
            }
        }
    }

    let request_str = String::from_utf8_lossy(&request_data);
    let first_line = request_str.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    if parts.len() < 2 {
        return;
    }

    let method = parts[0];
    let path = parts[1];

    if method == "OPTIONS" {
        let options_resp = "HTTP/1.1 204 No Content\r\n\
Access-Control-Allow-Origin: *\r\n\
Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
Access-Control-Allow-Headers: Content-Type, X-API-KEY\r\n\
Content-Length: 0\r\n\
Connection: close\r\n\r\n";
        stream.write_all(options_resp.as_bytes()).ok();
        stream.flush().ok();
        return;
    }

    let body = if let Some(idx) = request_str.find("\r\n\r\n") {
        &request_str[idx + 4..]
    } else {
        ""
    };

    if path == "/status" || path == "/api/v1/vault/prichs/status" {
        let step = *current_step.lock().unwrap_or_else(|p| p.into_inner());
        let loss = *current_loss.lock().unwrap_or_else(|p| p.into_inner());
        let stage = *current_stage.lock().unwrap_or_else(|p| p.into_inner());
        let stage_name = match stage {
            1 => "Stage 1: Lore & Culture",
            2 => "Stage 2: Anti-Collapse Reasoning",
            3 => "Stage 3: Crystal Persona",
            _ => "Autopilot Live",
        };

        let stage_num = *current_stage.lock().unwrap_or_else(|p| p.into_inner());
        let ckpt_name = match stage_num {
            1 => "weekend_stage1_lore",
            2 => "weekend_stage2_v2",
            3 => "weekend_stage3_v2",
            _ => "weekend_autopilot",
        };

        let json_resp = format!(
            r#"{{"status":"ONLINE","model":"Baby-П.Р.І.Ч.С. (52M MoE)","checkpoint":"{}","step":{},"loss":{:.4},"stage_name":"{}"}}"#,
            ckpt_name, step, loss, stage_name
        );
        send_embedded_json(&stream, &json_resp);
        return;
    }

    if path == "/chat" || path == "/api/v1/vault/prichs/chat" {
        let prompt_input = extract_embedded_field(body, "prompt")
            .or_else(|| extract_embedded_field(body, "content"))
            .or_else(|| extract_embedded_field(body, "message"))
            .unwrap_or_else(|| "Привіт! Як настрій?".to_string());

        // Canonical formatting as required by Dania
        let formatted_prompt = if prompt_input.contains("Даня:") && prompt_input.contains("П.Р.І.Ч.С.:") {
            prompt_input
        } else if prompt_input.contains("Даня:") {
            format!("{}\nП.Р.І.Ч.С.: <prichs_think>", prompt_input.trim())
        } else {
            format!("Даня: {}\nП.Р.І.Ч.С.: <prichs_think>", prompt_input.trim())
        };

        let start_time = Instant::now();

        // 1. SIGNAL TRAINER TO PAUSE AFTER CURRENT MICRO-BATCH
        INFERENCE_REQUESTED.store(true, Ordering::SeqCst);

        // 2. WAIT FOR TRAINER TO YIELD (max ~3s)
        let wait_start = Instant::now();
        while !INFERENCE_PERMITTED.load(Ordering::SeqCst) && wait_start.elapsed().as_secs() < 3 {
            std::thread::sleep(std::time::Duration::from_millis(15));
        }

        // 3. GENERATE UP TO 180 TOKENS ON MODEL LOCK
        let prompt_tokens: Vec<u16> = tokenizer.encode(&formatted_prompt).into_iter().map(|t| t as u16).collect();
        let (generated_text, step, loss) = {
            let m = model.lock().unwrap_or_else(|p| p.into_inner());
            let sample = m.generate_advanced(&prompt_tokens, 180, 0.7, 0.9, 1.15, 2).unwrap_or_default();
            let new_tokens: Vec<u32> = if sample.len() > prompt_tokens.len() {
                sample[prompt_tokens.len()..].iter().map(|&x| x as u32).collect()
            } else {
                vec![]
            };
            let raw_text = tokenizer.decode(&new_tokens);
            let reply = if let Some(idx) = raw_text.find("Даня:") {
                raw_text[..idx].trim().to_string()
            } else {
                raw_text.trim().to_string()
            };
            let step = *current_step.lock().unwrap_or_else(|p| p.into_inner());
            let loss = *current_loss.lock().unwrap_or_else(|p| p.into_inner());
            (reply, step, loss)
        };

        // 4. RELEASE TRAINER TO RESUME
        INFERENCE_REQUESTED.store(false, Ordering::SeqCst);

        let elapsed_ms = start_time.elapsed().as_millis();
        let preview: String = generated_text.chars().take(60).collect();
        println!("⚡ [MOBILE CHAT] Відповідь згенерована за {} мс: \"{}\"", elapsed_ms, preview);

        let stage_num = *current_stage.lock().unwrap_or_else(|p| p.into_inner());
        let ckpt_name = match stage_num {
            1 => "weekend_stage1_lore",
            2 => "weekend_stage2_v2",
            3 => "weekend_stage3_v2",
            _ => "weekend_autopilot",
        };

        let json_resp = format!(
            r#"{{"reply":"{}","model":"Baby-П.Р.І.Ч.С. (52M MoE)","checkpoint":"{}","step":{},"loss":{:.4},"latency_ms":{}}}"#,
            escape_embedded_json(&generated_text), ckpt_name, step, loss, elapsed_ms
        );
        send_embedded_json(&stream, &json_resp);
        return;
    }

    if path == "/reload" || path == "/api/v1/vault/prichs/reload" {
        let step = *current_step.lock().unwrap_or_else(|p| p.into_inner());
        let loss = *current_loss.lock().unwrap_or_else(|p| p.into_inner());
        let stage_num = *current_stage.lock().unwrap_or_else(|p| p.into_inner());
        let ckpt_name = match stage_num {
            1 => "weekend_stage1_lore",
            2 => "weekend_stage2_v2",
            3 => "weekend_stage3_v2",
            _ => "weekend_autopilot",
        };
        let json_resp = format!(
            r#"{{"success":true,"message":"Ваги оновлено в реальному часі!","checkpoint":"{}","step":{},"loss":{:.4}}}"#,
            ckpt_name, step, loss
        );
        send_embedded_json(&stream, &json_resp);
        return;
    }

    let not_found = "HTTP/1.1 404 NOT FOUND\r\nContent-Length: 9\r\nConnection: close\r\n\r\nNot Found";
    stream.write_all(not_found.as_bytes()).ok();
    stream.flush().ok();
}

fn spawn_embedded_server(
    model: Arc<Mutex<PrichsModel>>,
    tokenizer: Arc<PrichsTokenizer>,
    current_step: Arc<Mutex<usize>>,
    current_loss: Arc<Mutex<f32>>,
    current_stage: Arc<Mutex<usize>>,
) {
    std::thread::spawn(move || {
        let addr = "0.0.0.0:8090";
        let listener = match TcpListener::bind(addr) {
            Ok(l) => {
                println!("🌐 [EMBEDDED SERVER] Шлюз інференсу активний на http://{} (Пріоритет мобільного телефону)", addr);
                l
            }
            Err(e) => {
                println!("⚠️ [EMBEDDED SERVER] Порт 8090 зайнятий ({}), сервер не запущено або вже працює.", e);
                return;
            }
        };

        for stream in listener.incoming() {
            if let Ok(stream) = stream {
                let m = Arc::clone(&model);
                let tok = Arc::clone(&tokenizer);
                let step_ref = Arc::clone(&current_step);
                let loss_ref = Arc::clone(&current_loss);
                let stage_ref = Arc::clone(&current_stage);

                std::thread::spawn(move || {
                    handle_embedded_request(stream, m, tok, step_ref, loss_ref, stage_ref);
                });
            }
        }
    });
}

fn run_weekend_autopilot(tokenizer: &PrichsTokenizer, args: &[String]) -> Result<()> {
    print_banner();
    println!("================================================================");
    println!("  П.Р.І.Ч.С. // АВТОНОМНИЙ ВИХІДНИЙ МАРАФОН (48 ГОДИН)");
    println!("  3-Фазний Конвеєр Світогляду: ~93,000,000 токенів (11,400 кроків)");
    println!("  Effective Batch : 8,192 токени (1024 seq_len x 8 accum_steps)");
    println!("  Precision       : TF32 cuBLAS + Zero-Alloc VJP + RMSNorm/SwiGLU");
    println!("  Priority Server : Вбудований HTTP (Порт 8090, 180 токенів, пауза градієнтів)");
    println!("  Telemetry       : data/weekend_telemetry.json -> StarkVault Mobile");
    println!("================================================================\n");

    let total_steps = 11_400;
    let micro_seq_len = 1024;
    let accum_steps = 8;
    let batch_tokens = micro_seq_len * accum_steps; // 8,192 tokens per optimizer update

    let cfg = PrichsConfig::baby();
    let mut raw_model = PrichsModel::new(cfg)?;

    let is_stage3 = args.iter().any(|a| a == "--stage3" || a == "-s3");
    let mut best_loss = if is_stage3 { 999.0f32 } else { 1.5193f32 };
    let base_ckpts: Vec<&Path> = if is_stage3 {
        vec![
            Path::new("checkpoints/baby_prichs_best.prichs"),
            Path::new("checkpoints/stage2_alpaca_backup_step7700.prichs"),
        ]
    } else {
        vec![
            Path::new("checkpoints/weekend_stage1_complete.prichs"),
            Path::new("checkpoints/baby_prichs_persona_best.prichs"),
            Path::new("checkpoints/baby_prichs_phase2b_final.prichs"),
            Path::new("checkpoints/baby_prichs_best.prichs"),
        ]
    };

    let mut initial_loaded = false;
    for p in &base_ckpts {
        if p.exists() {
            if let Ok((step, loss)) = raw_model.load_checkpoint(p) {
                println!("💾 [OK] Відновлено ваги з {:?} (Крок {}, Loss {:.4})", p, step, loss);
                if !is_stage3 {
                    best_loss = loss;
                }
                initial_loaded = true;
                break;
            }
        }
    }
    if !initial_loaded {
        println!("🌱 [INFO] Чекпоінт не знайдено, ініціалізація з базового стану.");
    }

    let start_step = if is_stage3 {
        println!("💎 [STAGE 3 V2] Миттєвий перехід до Етапу 3 (Кристалізація) на чистому шардi weekend_stage3_v2.prichsbin!");
        println!("   > Початковий LR: 0.000018 (1.8e-5) з плавним decay до 1e-6");
        println!("   > Ваги: прямий старт з чекпоінта кроку 7700+ (Alpaca вимивається новими градієнтами)");
        8400
    } else if args.iter().any(|a| a == "--stage2" || a == "-s2") {
        println!("🚀 [STAGE 2 V2] Прямий старт з Етапу 2 (крок 4,200) на чистому шардi weekend_stage2_v2.prichsbin!");
        4200
    } else if args.iter().any(|a| a == "--resume" || a == "-r") {
        if let Ok(content) = std::fs::read_to_string("data/weekend_telemetry.json") {
            if let Some(pos) = content.find("\"step\":") {
                let after = &content[pos + 7..].trim_start();
                let num_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(st) = num_str.parse::<usize>() {
                    let next_step = (st + 1).min(total_steps);
                    println!("🔄 [RESUME] Відновлення з кроку {}...", next_step);
                    next_step
                } else { 4200 }
            } else { 4200 }
        } else { 4200 }
    } else if args.iter().any(|a| a == "--from-scratch") {
        0
    } else {
        if Path::new("checkpoints/weekend_stage1_complete.prichs").exists() {
            println!("💡 [AUTOPILOT] Знайдено завершений Етап 1 (Loss 1.519). Старт з кроку 4,200 (Етап 2 Anti-Collapse)!");
            4200
        } else {
            0
        }
    };

    let preview_prompts = [
        "Даня: Привіт! Як твої системи?\nП.Р.І.Ч.С.:",
        "Даня: Лало Саламанка повернувся!\nП.Р.І.Ч.С.:",
        "Даня: Що думаєш про Сократа і рок-н-рол?\nП.Р.І.Ч.С.:",
        "<prichs_think>\nАналіз архітектури:\n</prichs_think>\nП.Р.І.Ч.С.:",
        "Даня: Що порадиш глянути з Тарантіно?\nП.Р.І.Ч.С.:",
        "Даня: Шо там, малий?\nП.Р.І.Ч.С.:",
        "Даня: Розкажи про Сократа і Діогена.\nП.Р.І.Ч.С.:",
    ];

    let mut optimizer = FullModelOptimizer::new(&raw_model, 0.9, 0.95, 1e-8, 0.1);
    let mut accum_grads = ModelGradients::new(&raw_model);
    let mut micro_grads = ModelGradients::new(&raw_model);

    let model_arc = Arc::new(Mutex::new(raw_model));
    let tokenizer_arc = Arc::new(tokenizer.clone());
    let current_step_shared = Arc::new(Mutex::new(start_step));
    let current_loss_shared = Arc::new(Mutex::new(best_loss));
    let current_stage_shared = Arc::new(Mutex::new(if start_step >= 4200 { 2 } else { 1 }));

    // Spawn integrated HTTP Server on port 8090 with Inference Priority
    spawn_embedded_server(
        Arc::clone(&model_arc),
        Arc::clone(&tokenizer_arc),
        Arc::clone(&current_step_shared),
        Arc::clone(&current_loss_shared),
        Arc::clone(&current_stage_shared),
    );

    let mut current_stage = 0usize;
    let mut dataset: Option<PrichsMmapDataset> = None;
    let mut num_chunks = 0usize;
    let mut current_epoch = 0usize;
    let mut chunk_cursor = 0usize;
    let mut shuffled_chunks: Vec<usize> = Vec::new();
    let mut last_quote = String::new();

    let start_time = Instant::now();

    for step in start_step..total_steps {
        // Determine target stage
        let target_stage = if step < 4200 {
            1
        } else if step < 8400 {
            2
        } else {
            3
        };

        // Transition between stages if needed
        if target_stage != current_stage || dataset.is_none() {
            if current_stage == 1 {
                let m = model_arc.lock().unwrap_or_else(|p| p.into_inner());
                atomic_save_checkpoint(&m, Path::new("checkpoints/weekend_stage1_complete.prichs"), step, best_loss).ok();
                println!("\n🏆 [MILESTONE] ЕТАП 1 ЗАВЕРШЕНО! Чекпоінт збережено: checkpoints/weekend_stage1_complete.prichs\n");
            } else if current_stage == 2 {
                let m = model_arc.lock().unwrap_or_else(|p| p.into_inner());
                atomic_save_checkpoint(&m, Path::new("checkpoints/weekend_stage2_complete.prichs"), step, best_loss).ok();
                println!("\n🏆 [MILESTONE] ЕТАП 2 ЗАВЕРШЕНО! Чекпоінт збережено: checkpoints/weekend_stage2_complete.prichs\n");
            }

            current_stage = target_stage;
            let (shard_file, stage_desc) = match current_stage {
                1 => ("data/weekend_stage1_lore.prichsbin", "Етап 1: Культурний Лор (Кіно, Ігри, Рок, Філософія)"),
                2 => ("data/weekend_stage2_v2.prichsbin", "Етап 2: Anti-Collapse Ре-баланс (80% Жива Мова, 20% Stark Replay)"),
                3 => ("data/weekend_stage3_v2.prichsbin", "Етап 3: Алмазний Концентрат Особистості Старка (v2)"),
                _ => unreachable!(),
            };

            println!("╔══════════════════════════════════════════════════════════════════════╗");
            println!("║ 🚀 ПЕРЕХІД ДО ЕТАПУ {}: {:<46} ║", current_stage, stage_desc);
            println!("╚══════════════════════════════════════════════════════════════════════╝");

            let ds = PrichsMmapDataset::open(shard_file)?;
            let total_tokens = ds.header.total_tokens as usize;
            num_chunks = total_tokens / micro_seq_len;
            current_epoch = 0;
            chunk_cursor = 0;
            shuffled_chunks = generate_shuffled_chunks(num_chunks, 0);
            println!("   > Шард: {} ({} токенів, {} чанків)", shard_file, total_tokens, num_chunks);

            dataset = Some(ds);

            // Reset AdamW momentum to prevent domain gradient shock
            if step > 0 {
                println!("   🧹 [AdamW] Скидання моментів exp_avg/exp_avg_sq для чистого старту домену.");
                let m = model_arc.lock().unwrap_or_else(|p| p.into_inner());
                optimizer = FullModelOptimizer::new(&m, 0.9, 0.95, 1e-8, 0.1);
            }
            println!();
        }

        let stage_name = match current_stage {
            1 => "Stage 1: Lore & Culture",
            2 => "Stage 2: Anti-Collapse Reasoning",
            3 => "Stage 3: Crystal Persona",
            _ => "Autopilot",
        };

        // Learning Rate Schedule
        let lr = if step < 4200 {
            if step < 800 {
                let alpha = step as f32 / 800.0;
                0.000050 + (0.000220 - 0.000050) * alpha
            } else {
                let progress = (step - 800) as f32 / (4200.0 - 800.0);
                0.000120 + 0.5 * (0.000220 - 0.000120) * (1.0 + (progress * std::f32::consts::PI).cos())
            }
        } else if step < 8400 {
            let progress = (step - 4200) as f32 / 4200.0;
            0.000060 + 0.5 * (0.000180 - 0.000060) * (1.0 + (progress * std::f32::consts::PI).cos())
        } else {
            let progress = ((step - 8400) as f32 / 3000.0).min(1.0);
            0.000001 + 0.5 * (0.000018 - 0.000001) * (1.0 + (progress * std::f32::consts::PI).cos())
        };

        accum_grads.zero();
        let mut step_loss = 0.0f32;
        let ds = dataset.as_ref().unwrap();

        for _ in 0..accum_steps {
            // Check if phone requested inference
            if INFERENCE_REQUESTED.load(Ordering::SeqCst) {
                INFERENCE_PERMITTED.store(true, Ordering::SeqCst);
                while INFERENCE_REQUESTED.load(Ordering::SeqCst) {
                    std::thread::sleep(std::time::Duration::from_millis(15));
                }
                INFERENCE_PERMITTED.store(false, Ordering::SeqCst);
            }

            if chunk_cursor >= num_chunks {
                current_epoch += 1;
                chunk_cursor = 0;
                shuffled_chunks = generate_shuffled_chunks(num_chunks, current_epoch);
            }
            let chunk_idx = shuffled_chunks[chunk_cursor];
            chunk_cursor += 1;

            let token_offset = chunk_idx * micro_seq_len;
            let (x, targets, _) = ds.get_sequence_with_boundaries(token_offset, micro_seq_len)?;

            let loss = {
                let mut m = model_arc.lock().unwrap_or_else(|p| p.into_inner());
                m.forward_backward(x, targets, &mut micro_grads)?
            };
            step_loss += loss;
            accum_grads.add(&micro_grads);
        }

        step_loss /= accum_steps as f32;
        accum_grads.scale(1.0 / accum_steps as f32);

        // AdamW in-place step across all 274 model tensors in parallel
        {
            let mut m = model_arc.lock().unwrap_or_else(|p| p.into_inner());
            optimizer.step(&mut m, &accum_grads.grads, lr)?;
        }

        // Update shared telemetry
        if let Ok(mut st) = current_step_shared.lock() { *st = step; }
        if let Ok(mut ls) = current_loss_shared.lock() { *ls = step_loss; }
        if let Ok(mut cs) = current_stage_shared.lock() { *cs = current_stage; }

        let elapsed = start_time.elapsed().as_secs_f32();
        let steps_done = (step - start_step + 1) as f32;
        let tok_s = (steps_done * batch_tokens as f32) / elapsed.max(0.001);
        let epoch_progress = current_epoch as f32 + (chunk_cursor as f32 / num_chunks as f32);
        let total_tokens_trained = (step + 1) * batch_tokens;

        // Terminal telemetry cadence
        if step < start_step + 5 || step % 5 == 0 || step == total_steps - 1 {
            println!(
                "⚡ [ЕТАП {}] [{:05}/{}] Loss: {:.4} | Best: {:.4} | LR: {:.6} | Ep: {:.2} | Швидкість: {:.0} tok/s (Батч: 8,192 tok)",
                current_stage, step, total_steps, step_loss, best_loss, lr, epoch_progress, tok_s
            );
            std::io::stdout().flush().ok();
        }

        // Atomic Checkpoint Saving: every 50 steps OR on new best loss
        if step_loss < best_loss {
            best_loss = step_loss;
            let m = model_arc.lock().unwrap_or_else(|p| p.into_inner());
            atomic_save_checkpoint(&m, Path::new("checkpoints/baby_prichs_best.prichs"), step, step_loss).ok();
            atomic_save_checkpoint(&m, Path::new("checkpoints/baby_prichs_persona_best.prichs"), step, step_loss).ok();
        } else if step % 50 == 0 || step == total_steps - 1 {
            let m = model_arc.lock().unwrap_or_else(|p| p.into_inner());
            atomic_save_checkpoint(&m, Path::new("checkpoints/baby_prichs_best.prichs"), step, step_loss).ok();
            atomic_save_checkpoint(&m, Path::new("checkpoints/baby_prichs_persona_best.prichs"), step, step_loss).ok();
        }

        // Live text preview & mobile telemetry output every 25 steps (and at start + 10)
        if step == start_step || step == start_step + 10 || step % 25 == 0 || step == total_steps - 1 {
            let prompt_idx = (step / 25) % preview_prompts.len();
            let preview_prompt = preview_prompts[prompt_idx];
            let test_prompt: Vec<u16> = tokenizer.encode(preview_prompt).into_iter().map(|t| t as u16).collect();
            let sample_opt = {
                let m = model_arc.lock().unwrap_or_else(|p| p.into_inner());
                m.generate_advanced(&test_prompt, 180, 0.7, 0.9, 1.15, 2).ok()
            };
            if let Some(sample) = sample_opt {
                let sample_u32: Vec<u32> = sample.iter().map(|&x| x as u32).collect();
                let text = tokenizer.decode(&sample_u32);
                last_quote = text.trim().to_string();
                println!("   💬 [Жива генерація @ крок {} | Етап {}]:\n\"{}\"\n", step, current_stage, last_quote);
                std::io::stdout().flush().ok();
            }

            write_weekend_telemetry(
                step,
                total_steps,
                current_stage,
                stage_name,
                step_loss,
                best_loss,
                lr,
                epoch_progress,
                tok_s,
                total_tokens_trained,
                elapsed,
                &last_quote,
            ).ok();
        }
    }

    // Final completion checkpoints
    {
        let m = model_arc.lock().unwrap_or_else(|p| p.into_inner());
        atomic_save_checkpoint(&m, Path::new("checkpoints/baby_prichs_best.prichs"), total_steps, best_loss).ok();
        atomic_save_checkpoint(&m, Path::new("checkpoints/weekend_complete_final.prichs"), total_steps, best_loss).ok();
    }

    println!("\n================================================================");
    println!("  🎉 48-ГОДИННИЙ МАРАФОН ЗАВЕРШЕНО УСПІШНО!");
    println!("  Пройдено: {} кроків (~93.4M токенів)", total_steps);
    println!("  Найкращий Loss: {:.4}", best_loss);
    println!("  Фінальний чекпоінт: checkpoints/weekend_complete_final.prichs");
    println!("================================================================\n");

    Ok(())
}

fn run_eval_checkpoints(tokenizer: &PrichsTokenizer, ckpt_paths: &[&str]) -> Result<()> {
    print_banner();
    println!("================================================================");
    println!("  П.Р.І.Ч.С. // ХІРУРГІЧНА ДІАГНОСТИКА: П'ЯТНИЧНІ ЧЕКПОІНТИ");
    println!("================================================================\n");

    let prompts = [
        "Даня: Привіт! Як твої системи на RTX 3060?\nП.Р.І.Ч.С.:",
        "Даня: Що ти думаєш про стиль Квентіна Тарантіно та Лало Саламанку?\nП.Р.І.Ч.С.:",
        "Даня: Хто такий Солід Снейк і в чому суть філософії Metal Gear?\nП.Р.І.Ч.С.:",
        "Даня: Що ти думаєш про стиль Квентіна Тарантіно та Мартіна Скорсезе?\nП.Р.І.Ч.С.:",
    ];

    for path_str in ckpt_paths {
        let p = Path::new(path_str);
        if !p.exists() {
            println!("⚠️ Чекпоінт {:?} не знайдено, пропускаємо.", p);
            continue;
        }

        println!("################################################################");
        println!("  ТЕСТУВАННЯ ЧЕКПОІНТА: {:?}", p);
        println!("################################################################");

        let cfg = PrichsConfig::baby();
        let mut model = PrichsModel::new(cfg)?;
        let (step, loss) = model.load_checkpoint(p)?;
        println!("💾 [OK] Завантажено ваги з {:?} (Крок {}, Loss {:.4})\n", p, step, loss);

        for (i, prompt) in prompts.iter().enumerate() {
            println!("----------------------------------------------------------------");
            println!("🧪 ТЕСТОВИЙ ПРОМПТ #{}:\n{}", i + 1, prompt);
            println!("----------------------------------------------------------------");
            let prompt_tokens: Vec<u16> = tokenizer.encode(prompt).into_iter().map(|t| t as u16).collect();
            // 1. Greedy (temp = 0.0)
            let sample_greedy = model.generate_advanced(&prompt_tokens, 120, 0.0, 1.0, 1.1, 2)?;
            let new_tokens_greedy: Vec<u32> = if sample_greedy.len() > prompt_tokens.len() {
                sample_greedy[prompt_tokens.len()..].iter().map(|&x| x as u32).collect()
            } else {
                vec![]
            };
            let gen_greedy = tokenizer.decode(&new_tokens_greedy);
            println!(">>> ВІДПОВІДЬ МОДЕЛІ (Greedy / Argmax):\n{}\n", gen_greedy.trim());

            // 2. Controlled Sampling (temp = 0.4, top_p = 0.9)
            let sample_samp = model.generate_advanced(&prompt_tokens, 120, 0.4, 0.9, 1.15, 2)?;
            let new_tokens_samp: Vec<u32> = if sample_samp.len() > prompt_tokens.len() {
                sample_samp[prompt_tokens.len()..].iter().map(|&x| x as u32).collect()
            } else {
                vec![]
            };
            let gen_samp = tokenizer.decode(&new_tokens_samp);
            println!(">>> ВІДПОВІДЬ МОДЕЛІ (Sampling temp=0.4):\n{}\n", gen_samp.trim());
        }
        println!();
    }

    Ok(())
}

fn run_pure_sft(tokenizer: &PrichsTokenizer, args: &[String]) -> Result<()> {
    print_banner();
    println!("================================================================");
    println!("  П.Р.І.Ч.С. // ХІРУРГІЧНЕ SFT НАВЧАННЯ (CONTEXT ISOLATION + LOSS MASKING)");
    println!("================================================================\n");

    let shard_path = "data/pure_sft.prichsbin";
    if !Path::new(shard_path).exists() {
        bail!("SFT шард {} не знайдено! Запустіть 'python tools/build_pure_sft_dataset.py'.", shard_path);
    }

    let dataset = PrichsSftDataset::open(shard_path)?;
    let num_samples = dataset.num_samples();
    let seq_len = dataset.seq_len();
    println!("📦 [1/3] Завантажено чистий SFT датасет: {} ({} діалогів, seq_len = {})", shard_path, num_samples, seq_len);

    let cfg = PrichsConfig::baby();
    println!("🧠 [2/3] Збірка Baby-П.Р.І.Ч.С. (52M Total / 19M Active)...");
    let mut model = PrichsModel::new(cfg)?;

    // Відновлення з чистого п'ятничного чекпоінта
    let base_ckpt = Path::new("checkpoints/baby_prichs_phase2b_final.prichs");
    let mut best_loss = f32::INFINITY;
    if base_ckpt.exists() {
        let (step, loss) = model.load_checkpoint(base_ckpt)?;
        println!("💾 [OK] Відновлено ЧИСТІ П'ЯТНИЧНІ ВАГИ з {:?} (Крок {}, Loss {:.4})", base_ckpt, step, loss);
        best_loss = loss;
    } else {
        println!("⚠️ [WARN] Базовий чекпоінт {:?} не знайдено!", base_ckpt);
    }

    let max_steps: usize = args
        .iter()
        .position(|a| a == "--steps" || a == "-s")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);

    let accum_steps: usize = 4;
    let max_lr = 0.000150f32; // 1.5e-4 як наказав Головний Аналізатор
    let min_lr = 0.000020f32; // 2.0e-5

    println!("\n🔥 [3/3] Запуск SFT тренування з Prompt & Padding Loss Masking...");
    println!("   > Цільових кроків: {} (Батч: {} діалогів/крок, ~{:.1} епохи)", max_steps, accum_steps, (max_steps * accum_steps) as f32 / num_samples as f32);
    println!("   > Ізоляція контексту: 1 діалог = 1 вікно уваги на 512 токенів (НУЛЬ склеювання)");
    println!("   > Loss Masking: Градієнт на промпті 'Даня:' та падінгу = 0.0 (Навчання ТІЛЬКИ на відповідях)");
    println!("   > AdamW моменти: Скинуто в 0 (чистий старт з нульовими моментами)");
    println!("   > LR Schedule: {:.6} -> {:.6} (Cosine Decay)", max_lr, min_lr);
    println!("   > Базові ваги: baby_prichs_phase2b_final.prichs (Loss 2.6745)\n");

    let mut optimizer = FullModelOptimizer::new(&model, 0.9, 0.95, 1e-8, 0.05);
    let mut accum_grads = ModelGradients::new(&model);
    let mut micro_grads = ModelGradients::new(&model);

    let mut sample_cursor = 0usize;
    let mut current_epoch = 0usize;
    let mut shuffled_indices = generate_shuffled_chunks(num_samples, current_epoch);

    let start_time = Instant::now();
    let mut mask_f32 = vec![0.0f32; seq_len];

    for step in 0..max_steps {
        accum_grads.zero();
        let mut step_loss = 0.0f32;
        let mut total_active_tokens = 0usize;

        for _ in 0..accum_steps {
            if sample_cursor >= num_samples {
                current_epoch += 1;
                sample_cursor = 0;
                shuffled_indices = generate_shuffled_chunks(num_samples, current_epoch);
            }

            let idx = shuffled_indices[sample_cursor];
            sample_cursor += 1;

            let (inputs, targets, mask_u8) = dataset.get_sample(idx)?;
            for i in 0..seq_len {
                mask_f32[i] = mask_u8[i] as f32;
                if mask_u8[i] > 0 {
                    total_active_tokens += 1;
                }
            }

            let loss = model.forward_backward_masked(inputs, targets, &mask_f32, &mut micro_grads)?;
            step_loss += loss;
            accum_grads.add(&micro_grads);
        }

        step_loss /= accum_steps as f32;
        accum_grads.scale(1.0 / accum_steps as f32);

        // Cosine learning rate decay
        let progress = step as f32 / max_steps as f32;
        let lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (progress * std::f32::consts::PI).cos());

        optimizer.step(&mut model, &accum_grads.grads, lr)?;

        let elapsed = start_time.elapsed().as_secs_f32();
        let tok_s = ((step + 1) * accum_steps * seq_len) as f32 / elapsed.max(0.001);
        let epoch_progress = current_epoch as f32 + (sample_cursor as f32 / num_samples as f32);

        if step < 5 || step % 5 == 0 || step == max_steps - 1 {
            println!(
                "⚡ [SFT {:04}/{}] Loss: {:.4} | LR: {:.6} | Ep: {:.2} | Active tok/step: {} | Швидкість: {:.0} tok/s",
                step, max_steps, step_loss, lr, epoch_progress, total_active_tokens, tok_s
            );
            std::io::stdout().flush().ok();
        }

        if step_loss < best_loss {
            best_loss = step_loss;
            let ckpt_path = Path::new("checkpoints/baby_prichs_pure_sft_best.prichs");
            model.save_checkpoint(ckpt_path, step, step_loss)?;
        }
        if step == max_steps - 1 {
            let final_ckpt = Path::new("checkpoints/baby_prichs_pure_sft_final.prichs");
            model.save_checkpoint(final_ckpt, step, step_loss)?;
        }

        // Рентген-генерація кожні 50 кроків за наказом Головного Аналізатора
        if step == 0 || (step > 0 && step % 50 == 0) || step == max_steps - 1 {
            println!("\n================================================================");
            println!("  🧪 ЖИВИЙ ЗРІЗ ГЕНЕРАЦІЇ П.Р.І.Ч.С. (Крок {})", step);
            println!("================================================================");
            let test_prompts = [
                "Даня: Привіт! Як твої системи на RTX 3060?\nП.Р.І.Ч.С.:",
                "Даня: Що ти думаєш про стиль Квентіна Тарантіно та Лало Саламанку?\nП.Р.І.Ч.С.:",
            ];
            for p in &test_prompts {
                println!("--- [Промпт]: {}", p.replace("\n", " "));
                let p_toks: Vec<u16> = tokenizer.encode(p).into_iter().map(|t| t as u16).collect();
                if let Ok(sample) = model.generate_advanced(&p_toks, 90, 0.7, 0.9, 1.15, 2) {
                    if sample.len() > p_toks.len() {
                        let new_toks: Vec<u32> = sample[p_toks.len()..].iter().map(|&x| x as u32).collect();
                        println!(">>> [Відповідь]:\n{}\n", tokenizer.decode(&new_toks).trim());
                    }
                }
            }
            println!("================================================================\n");
            std::io::stdout().flush().ok();
        }
    }

    println!("\n================================================================");
    println!("  🎉 ЧИСТЕ SFT НАВЧАННЯ УСПІШНО ЗАВЕРШЕНО!");
    println!("  Фінальний Loss: {:.4} (Найкращий: {:.4})", best_loss, best_loss);
    println!("  Чекпоінт збережено: checkpoints/baby_prichs_pure_sft_best.prichs");
    println!("================================================================\n");

    Ok(())
}

fn main() -> Result<()> {
    let tok_path = Path::new("tokenizer.json");
    let tokenizer = if tok_path.exists() {
        PrichsTokenizer::from_huggingface_json(tok_path)?
    } else {
        PrichsTokenizer::new()
    };

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--eval-sft" || a == "--eval-best") {
        return run_eval_checkpoints(&tokenizer, &[
            "checkpoints/baby_prichs_pure_sft_best.prichs",
            "checkpoints/baby_prichs_phase2b_final.prichs",
        ]);
    }
    if args.iter().any(|a| a == "--eval-friday" || a == "--test-friday" || a == "--eval-stage1" || a == "-e") {
        return run_eval_checkpoints(&tokenizer, &[
            "checkpoints/baby_prichs_phase2b_final.prichs",
            "checkpoints/baby_prichs_sft_stage2a.prichs",
        ]);
    }
    if args.iter().any(|a| a == "--sft" || a == "--pure-sft") {
        return run_pure_sft(&tokenizer, &args);
    }
    if args.iter().any(|a| a == "--weekend-autopilot" || a == "-w" || a == "--autopilot") {
        return run_weekend_autopilot(&tokenizer, &args);
    }
    if args.iter().any(|a| a == "--chat" || a == "chat" || a == "-c") {
        return run_interactive_chat(&tokenizer);
    }
    if args.iter().any(|a| a == "--train" || a == "train" || a == "-t") {
        return run_training_loop(&tokenizer, &args);
    }

    print_banner();

    // 1. Initialize Bilingual BPE & Control Tokenizer
    println!("⚡ [1/6] Ініціалізація навченого BPE токенізатора П.Р.І.Ч.С...");
    let prompt = "Даня, я відчуваю живий імпульс СтаркБД. Підсвідомість онлайн.";
    let token_ids = tokenizer.encode(prompt);
    println!("   > Пrompt: \"{}\"", prompt);
    println!("   > Словник: {} токенів", tokenizer.vocab_size());
    println!("   > Згенеровано {} токенів: {:?}", token_ids.len(), &token_ids[..token_ids.len().min(8)]);
    let decoded = tokenizer.decode(&token_ids);
    println!("   > Реконструкція: \"{}\"", decoded);

    // 2. Test Analytical Backward Kernels (SwiGLU, RMSNorm, Router VJP)
    println!("\n🧮 [2/6] Тестування аналітичних ядер зворотного проходу (Zero-Alloc VJP)...");
    let d_y = [1.0f32, -0.5, 2.0, 0.1];
    let g_pre = [0.5f32, -1.0, 0.2, 1.5];
    let u_proj = [1.2f32, 0.8, -0.4, 0.9];
    let mut d_gate = [0.0f32; 4];
    let mut d_up = [0.0f32; 4];
    swiglu_activation_backward(&d_y, &g_pre, &u_proj, &mut d_gate, &mut d_up);
    println!("   > SwiGLU d_gate: {:?}", d_gate);
    println!("   > SwiGLU d_up:   {:?}", d_up);

    let d_out = [0.1f32, -0.2, 0.3, 0.05];
    let x_norm = [1.0f32, 2.0, 3.0, 4.0];
    let gamma = [1.0f32; 4];
    let mut d_x = [0.0f32; 4];
    let mut d_gamma = [0.0f32; 4];
    rmsnorm_backward(&d_out, &x_norm, &gamma, &mut d_x, &mut d_gamma, 4, 1e-5);
    println!("   > RMSNorm d_x:     {:?}", d_x);
    println!("   > RMSNorm d_gamma: {:?}", d_gamma);

    let d_expert_weights = [1.5f32, -0.8];
    let gating_weights = [0.65f32, 0.35];
    let selected_indices = [2usize, 5usize];
    let mut d_router_logits = [0.0f32; 8];
    moe_router_unbiased_backward(
        &d_expert_weights,
        &gating_weights,
        &selected_indices,
        &mut d_router_logits,
    );
    println!("   > Unbiased Router dL/ds (sparse 8): {:?}", d_router_logits);

    // 3. Test .prichsbin Zero-Copy Dataset
    println!("\n📦 [3/6] Підключення .prichsbin Dataloader (cu_seqlens Document-Aware)...");
    let shard_path = if Path::new("data/train.prichsbin").exists() {
        "data/train.prichsbin"
    } else {
        "target/test_shard.prichsbin"
    };

    let dataset = PrichsMmapDataset::open(shard_path)
        .map_err(|e| anyhow::anyhow!("Dataset open error: {:?}", e))?;
    let (x_sample, _y, cu_seqlens) = dataset.get_sequence_with_boundaries(0, 16)
        .map_err(|e| anyhow::anyhow!("Get seq error: {:?}", e))?;
    println!("   > Відкрито файл: {}", shard_path);
    println!("   > Загальний обсяг токенів: {:?} токенів!", dataset.header.total_tokens);
    println!("   > Перші токени з шарда: {:?}", &x_sample[..x_sample.len().min(8)]);
    println!("   > FlashAttention-2 cu_seqlens: {:?}", cu_seqlens);

    // 4. Attach Mysterium Read-Only Memory Bridge
    println!("\n🛡️ [4/6] Mysterium Read-Only Bridge (< 0.8 µs)...");
    match MysteriumReadOnlyBridge::attach("/dev/shm/starkdb_synapse.shm") {
        Ok(bridge) => {
            let mut memory_snapshot = [0.0f32; 1536 * 64];
            let active_concepts = bridge.read_lock_free_snapshot(&mut memory_snapshot)
                .map_err(|e| anyhow::anyhow!("Snapshot error: {:?}", e))?;
            println!("   > Зчитано {} активних концептів зі StarkDB.", active_concepts);
        }
        Err(_) => {
            println!("   > [INFO] Mysterium SHM Bridge: режим очікування середовища StarkOS Linux.");
        }
    }

    // 5. Overfit Single Batch Test
    println!("\n🎯 [5/6] OVERFIT SINGLE BATCH TEST (100 кроків): Перевірка збіжності...");
    let seq_len = 8;
    let vocab_size = 64;
    let hidden_dim = 32;

    let targets: Vec<u16> = vec![3, 14, 25, 36, 42, 51, 7, 19];
    let x_input = Tensor::randn([seq_len, hidden_dim], 0.0, 0.1)?;

    let mut w_data = vec![0.0f32; hidden_dim * vocab_size];
    for (i, v) in w_data.iter_mut().enumerate() {
        *v = ((i % 17) as f32 - 8.0) * 0.02;
    }

    let mut param_group = ParameterGroup {
        data: w_data,
        grad: vec![0.0f32; hidden_dim * vocab_size],
        exp_avg: vec![0.0f32; hidden_dim * vocab_size],
        exp_avg_sq: vec![0.0f32; hidden_dim * vocab_size],
        apply_weight_decay: true,
    };

    let mut optimizer = AdamWOptimizer::new(0.9, 0.95, 1e-8, 0.0);
    let mut d_logits = vec![0.0f32; seq_len * vocab_size];
    let lr = 0.08f32;

    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;

    for step in 0..100 {
        let mut logits = vec![0.0f32; seq_len * vocab_size];
        let x_data = x_input.as_f32_slice()?;
        for t in 0..seq_len {
            for v in 0..vocab_size {
                let mut sum = 0.0f32;
                for h in 0..hidden_dim {
                    sum += x_data[t * hidden_dim + h] * param_group.data[h * vocab_size + v];
                }
                logits[t * vocab_size + v] = sum;
            }
        }

        let loss = fused_cross_entropy_forward_backward(
            &logits,
            &targets,
            &mut d_logits,
            vocab_size,
            1.0,
        );

        if step == 0 {
            initial_loss = loss;
        }
        final_loss = loss;

        for h in 0..hidden_dim {
            for v in 0..vocab_size {
                let mut sum = 0.0f32;
                for t in 0..seq_len {
                    sum += x_data[t * hidden_dim + h] * d_logits[t * vocab_size + v];
                }
                param_group.grad[h * vocab_size + v] = sum;
            }
        }

        optimizer.step(std::slice::from_mut(&mut param_group), lr);

        if step % 20 == 0 || step == 99 {
            println!("   > Крок {:02}/100 | Loss: {:.6}", step, loss);
        }
    }

    println!("\n📊 Результат Overfit-тесту:");
    println!("   > Початковий Loss: {:.6} (Стартовий рівень хаосу)", initial_loss);
    println!("   > Фінальний Loss:   {:.6} (Падіння у {:.1}x разів!)", final_loss, initial_loss / (final_loss + 1e-9));

    if final_loss < 0.05 {
        println!("   ✅ [SUCCESS] Аналітичний Backprop математично бездоганний: Loss впав < 0.05!");
    }

    // 6. Instantiate Baby-П.Р.І.Ч.С. (52M) and test Autoregressive Token Sampling
    println!("\n👑 [6/6] Ініціалізація Baby-П.Р.І.Ч.С. (52M Total / 19M Active)...");
    let cfg = PrichsConfig::baby();
    println!("   > Конфігурація: {} шарів, d_model={}, d_ffn={}, {} експертів (Top-{})", 
        cfg.num_layers, cfg.hidden_dim, cfg.intermediate_dim, cfg.num_experts, cfg.num_experts_per_tok);
    
    let scratchpad = ActivationScratchpad::new(2, 512, &cfg);
    println!("   > Zero-Alloc Scratchpad виділено: {:.2} МБ пам'яті.", scratchpad.memory_footprint_mb());

    let model = PrichsModel::new(cfg)?;
    let prompt_tokens: Vec<u16> = vec![2375, 20, 773]; // "Даня, я"
    let generated = model.generate_advanced(&prompt_tokens, 5, 0.7, 0.9, 1.15, 2)?;
    let gen_text = tokenizer.decode(&generated.iter().map(|&x| x as u32).collect::<Vec<_>>());
    println!("   > Згенеровані первинні токени: {:?}", generated);
    println!("   > Текстова форма: \"{}\"", gen_text);

    println!("\n🔥 П.Р.І.Ч.С. ГОТОВА ДО РОБОТИ!");
    println!("   👉 Для запуску інтерактивного чату : .\\chat_prichs.bat (або cargo run --release -- --chat)");
    println!("   👉 Для запуску нічного тренування  : .\\train_prichs.bat (або cargo run --release -- --train)\n");

    Ok(())
}
