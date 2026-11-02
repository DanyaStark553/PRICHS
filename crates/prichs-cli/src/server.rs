use anyhow::Result;
use prichs_model::{PrichsConfig, PrichsModel};
use prichs_tokenizer::PrichsTokenizer;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

struct AppState {
    model: PrichsModel,
    tokenizer: PrichsTokenizer,
    current_step: usize,
    current_loss: f32,
    checkpoint_name: String,
}

fn load_best_checkpoint(model: &mut PrichsModel) -> (usize, f32, String) {
    let best_path = Path::new("checkpoints/baby_prichs_best.prichs");
    let persona_path = Path::new("checkpoints/baby_prichs_persona_best.prichs");

    let target_path = match (best_path.metadata(), persona_path.metadata()) {
        (Ok(m_best), Ok(m_persona)) => {
            let t_best = m_best.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            let t_persona = m_persona.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            if t_best >= t_persona { best_path } else { persona_path }
        }
        (Ok(_), Err(_)) => best_path,
        (Err(_), Ok(_)) => persona_path,
        _ => best_path,
    };

    if target_path.exists() {
        match model.load_checkpoint(target_path) {
            Ok((step, loss)) => {
                let name = target_path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "checkpoint.prichs".to_string());
                println!("💾 [OK] Завантажено ваги з {:?} (Крок {}, Loss {:.4})", target_path, step, loss);
                return (step, loss, name);
            }
            Err(e) => {
                println!("⚠️ [WARN] Помилка читання чекпоінта {:?}: {}", target_path, e);
            }
        }
    }
    (0, 0.0, "initial_weights".to_string())
}

fn extract_json_field(body: &str, field_name: &str) -> Option<String> {
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

fn extract_json_number(body: &str, field_name: &str) -> Option<f32> {
    let pattern = format!("\"{}\":", field_name);
    let pos = body.find(&pattern)?;
    let after = &body[pos + pattern.len()..].trim_start();
    let num_str: String = after.chars().take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-').collect();
    num_str.parse().ok()
}

fn escape_json(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn send_json(mut stream: &TcpStream, json: &str) -> Result<()> {
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
    stream.write_all(header.as_bytes())?;
    stream.write_all(bytes)?;
    stream.flush()?;
    Ok(())
}

fn handle_connection(mut stream: TcpStream, state: Arc<Mutex<AppState>>) -> Result<()> {
    let mut request_data = Vec::with_capacity(4096);
    let mut temp = [0u8; 2048];

    // Read full HTTP request including headers and body according to Content-Length
    loop {
        let n = stream.read(&mut temp)?;
        if n == 0 {
            break;
        }
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
        return Ok(());
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
        stream.write_all(options_resp.as_bytes())?;
        stream.flush()?;
        return Ok(());
    }

    let body = if let Some(idx) = request_str.find("\r\n\r\n") {
        &request_str[idx + 4..]
    } else {
        ""
    };

    if path == "/status" || path == "/api/v1/vault/prichs/status" {
        let app = state.lock().unwrap_or_else(|p| p.into_inner());
        let telemetry_content = std::fs::read_to_string("data/weekend_telemetry.json").ok();
        let (disp_step, disp_loss, stage_name, tok_s) = if let Some(ref t_json) = telemetry_content {
            let t_step = extract_json_number(t_json, "step").map(|n| n as usize).unwrap_or(app.current_step);
            let t_loss = extract_json_number(t_json, "loss").unwrap_or(app.current_loss);
            let s_name = extract_json_field(t_json, "stage_name").unwrap_or_else(|| "Live".to_string());
            let t_speed = extract_json_number(t_json, "tok_per_sec").unwrap_or(0.0);
            (t_step, t_loss, s_name, t_speed)
        } else {
            (app.current_step, app.current_loss, "Ready".to_string(), 0.0)
        };

        let json_resp = format!(
            r#"{{"status":"ONLINE","model":"Baby-П.Р.І.Ч.С. (52M MoE)","checkpoint":"{}","step":{},"loss":{:.4},"stage_name":"{}","tok_per_sec":{:.1}}}"#,
            app.checkpoint_name, disp_step, disp_loss, stage_name, tok_s
        );
        return send_json(&stream, &json_resp);
    }

    if path == "/reload" || path == "/api/v1/vault/prichs/reload" {
        let mut app = state.lock().unwrap_or_else(|p| p.into_inner());
        let (step, loss, name) = load_best_checkpoint(&mut app.model);
        app.current_step = step;
        app.current_loss = loss;
        app.checkpoint_name = name.clone();

        let json_resp = format!(
            r#"{{"success":true,"message":"Ваги оновлено успішно!","checkpoint":"{}","step":{},"loss":{:.4}}}"#,
            name, step, loss
        );
        return send_json(&stream, &json_resp);
    }

    if path == "/chat" || path == "/api/v1/vault/prichs/chat" {
        let prompt_input = extract_json_field(body, "prompt")
            .or_else(|| extract_json_field(body, "content"))
            .or_else(|| extract_json_field(body, "message"))
            .unwrap_or_else(|| "Привіт! Як почуваєшся?".to_string());

        let formatted_prompt = if prompt_input.contains("Даня:") && prompt_input.contains("П.Р.І.Ч.С.:") {
            prompt_input
        } else if prompt_input.contains("Даня:") {
            format!("{}\nП.Р.І.Ч.С.: <prichs_think>", prompt_input.trim())
        } else {
            format!("Даня: {}\nП.Р.І.Ч.С.: <prichs_think>", prompt_input.trim())
        };

        let start_time = Instant::now();
        let (generated_text, step, loss, ckpt) = {
            let app = state.lock().unwrap_or_else(|p| p.into_inner());
            let prompt_tokens: Vec<u16> = app.tokenizer.encode(&formatted_prompt).into_iter().map(|t| t as u16).collect();
            let sample = app.model.generate_advanced(&prompt_tokens, 180, 0.7, 0.9, 1.15, 2)?;
            let new_tokens: Vec<u32> = if sample.len() > prompt_tokens.len() {
                sample[prompt_tokens.len()..].iter().map(|&x| x as u32).collect()
            } else {
                vec![]
            };
            let raw_text = app.tokenizer.decode(&new_tokens);

            // Clean response: cut off if model hallucinated user turn
            let reply = if let Some(idx) = raw_text.find("Даня:") {
                raw_text[..idx].trim().to_string()
            } else {
                raw_text.trim().to_string()
            };
            (reply, app.current_step, app.current_loss, app.checkpoint_name.clone())
        };

        let elapsed_ms = start_time.elapsed().as_millis();
        let preview: String = generated_text.chars().take(60).collect();
        println!("⚡ [PrichsServer] Відповідь згенерована за {} мс: \"{}\"", elapsed_ms, preview);

        let json_resp = format!(
            r#"{{"reply":"{}","model":"Baby-П.Р.І.Ч.С. (52M MoE)","checkpoint":"{}","step":{},"loss":{:.4},"latency_ms":{}}}"#,
            escape_json(&generated_text), ckpt, step, loss, elapsed_ms
        );
        return send_json(&stream, &json_resp);
    }

    let not_found = "HTTP/1.1 404 NOT FOUND\r\nContent-Length: 9\r\nConnection: close\r\n\r\nNot Found";
    stream.write_all(not_found.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    println!("\x1b[38;2;56;189;248m");
    println!("================================================================");
    println!("  П.Р.І.Ч.С. // Sovereign Inference Server (Port 8090)          ");
    println!("  Architecture : Baby-П.Р.І.Ч.С. (52M Total / 19M Active MoE)   ");
    println!("  Endpoints    : /status | /chat | /reload                      ");
    println!("================================================================\x1b[0m\n");

    let tok_path = Path::new("tokenizer.json");
    let tokenizer = if tok_path.exists() {
        PrichsTokenizer::from_huggingface_json(tok_path)?
    } else {
        PrichsTokenizer::new()
    };

    let cfg = PrichsConfig::baby();
    let mut model = PrichsModel::new(cfg)?;
    let (step, loss, name) = load_best_checkpoint(&mut model);

    let state = Arc::new(Mutex::new(AppState {
        model,
        tokenizer,
        current_step: step,
        current_loss: loss,
        checkpoint_name: name,
    }));

    let addr = "0.0.0.0:8090";
    let listener = TcpListener::bind(addr)?;
    println!("🚀 [ONLINE] П.Р.І.Ч.С. Inference Server слухає на http://{}", addr);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let state_clone = Arc::clone(&state);
                std::thread::spawn(move || {
                    if let Err(e) = handle_connection(stream, state_clone) {
                        eprintln!("⚠️ [ERROR] Connection error: {}", e);
                    }
                });
            }
            Err(e) => {
                eprintln!("⚠️ [ERROR] Accept error: {}", e);
            }
        }
    }

    Ok(())
}
