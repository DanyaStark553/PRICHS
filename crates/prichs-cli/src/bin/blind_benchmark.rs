use anyhow::{Context, Result};
use prichs_model::{PrichsConfig, PrichsModel};
use prichs_tokenizer::PrichsTokenizer;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    println!("\x1b[38;2;244;63;94m");
    println!(r#"
  ██████╗  ██╗     ██╗███╗   ██╗██████╗     ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
  ██╔══██╗ ██║     ██║████╗  ██║██╔══██╗    ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
  ██████╔╝ ██║     ██║██╔██╗ ██║██║  ██║    ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
  ██╔══██╗ ██║     ██║██║╚██╗██║██║  ██║    ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
  ██████╔╝ ███████╗██║██║ ╚████║██████╔╝    ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
  ╚═════╝  ╚══════╝╚═╝╚═╝  ╚═══╝╚═════╝     ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝
  П. Р. І. Ч. С. v2 // Blind Benchmark Suite (10 Sectors x Calibrated Inference)
  Sampling: temp=0.20, top_k=30, top_p=0.85, rep_penalty=1.03
    "#);
    println!("\x1b[0m");

    let args: Vec<String> = std::env::args().collect();
    let default_ckpt = "checkpoints/v2_sft_best.prichs";
    let ckpt_path_str = args.get(1).map(|s| s.as_str()).unwrap_or(default_ckpt);
    let ckpt_path = Path::new(ckpt_path_str);

    if !ckpt_path.exists() {
        eprintln!("⚠️ Чекпоінт {:?} не знайдено!", ckpt_path);
        std::process::exit(1);
    }

    let tok_path = Path::new("tokenizer.json");
    let tokenizer = PrichsTokenizer::from_huggingface_json(tok_path)
        .context("Не вдалося завантажити токенізатор")?;

    let cfg = PrichsConfig::v2_moe();
    println!("⏳ Завантаження моделі з {:?}...", ckpt_path);
    let mut model = PrichsModel::new(cfg)?;
    let (step, loss) = model.load_checkpoint(ckpt_path)?;
    println!("✅ Модель завантажено: крок {}, збережений Loss: {:.4}", step, loss);

    let test_prompts = [
        (
            "1. Майстерня та зв'язок з Данею",
            "Даня: Прічс, третя ночі, я втомився. Як стан систем?\nП.Р.І.Ч.С.:"
        ),
        (
            "2. Апаратна діагностика та VRM",
            "Даня: Чому виникає перегрів VRM на RTX 3060?\nП.Р.І.Ч.С.:"
        ),
        (
            "3. Архітектура StarkOS & Rust/CUDA",
            "Даня: Як у StarkOS організовано спільну пам'ять без колізій банків?\nП.Р.І.Ч.С.:"
        ),
        (
            "4. Кінематограф: Тарантіно",
            "Даня: Поясни сцену в таверні з трьома пальцями в Безславних виродках.\nП.Р.І.Ч.С.:"
        ),
        (
            "5. Кінематограф: Лало Саламанка",
            "Даня: Чому Лало Саламанка такий небезпечний порівняно з іншими картельними босами?\nП.Р.І.Ч.С.:"
        ),
        (
            "6. Геймінг: Metal Gear Solid",
            "Даня: У чому головний філософський конфлікт між Біг Боссом і Зіро в MGS?\nП.Р.І.Ч.С.:"
        ),
        (
            "7. Філософія: Фрідріх Ніцше",
            "Даня: Що означає Amor Fati у Ніцше і як це застосувати до нашої боротьби?\nП.Р.І.Ч.С.:"
        ),
        (
            "8. Філософія: Діоген Синопський",
            "Даня: Чому Діоген ходив з ліхтарем удень і що він хотів цим довести?\nП.Р.І.Ч.С.:"
        ),
        (
            "9. Гараж: Audi Quattro & Турбо",
            "Даня: Як налаштувати турбонаддув на Audi Quattro 20V Turbo, щоб не порвало поршневу?\nП.Р.І.Ч.С.:"
        ),
        (
            "10. Тактичні кризові операції",
            "Даня: Прічс, тривога! Воркер спіймав дедлок на м'ютексі, шина перевантажена!\nП.Р.І.Ч.С.:"
        ),
    ];

    println!("\n{}", "=".repeat(80));
    println!("🎯 ЗАПУСК СЛІПОГО БЕНЧМАРКУ (10 СЕКТОРІВ)");
    println!("{}", "=".repeat(80));

    let eos_id = 2u16;

    for (idx, (title, prompt)) in test_prompts.iter().enumerate() {
        println!("\n--------------------------------------------------------------------------------");
        println!("📌 [{}] {}", idx + 1, title);
        println!("ВХІДНИЙ ПРОМПТ:\n{}", prompt);
        println!("--------------------------------------------------------------------------------");

        let prompt_ids: Vec<u16> = tokenizer.encode(prompt).into_iter().map(|id| id as u16).collect();
        let t_gen = Instant::now();

        // Calibrated sampling: temp=0.20, top_p=0.85, rep_penalty=1.03, max_tokens=140
        match model.generate_advanced(&prompt_ids, 140, 0.20, 0.85, 1.03, eos_id) {
            Ok(gen_ids) => {
                let gen_u32: Vec<u32> = gen_ids.into_iter().map(|id| id as u32).collect();
                let full_output = tokenizer.decode(&gen_u32);
                let dur = t_gen.elapsed();
                println!("ВІДПОВІДЬ П.Р.І.Ч.С. ({:.2?}):\n{}", dur, full_output.trim());
            }
            Err(e) => {
                println!("⚠️ Помилка генерації: {}", e);
            }
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("✅ СЛІПИЙ БЕНЧМАРК УСПІШНО ЗАВЕРШЕНО!");
    println!("{}", "=".repeat(80));

    Ok(())
}
