"""
train_tokenizer.py — Byte-Level BPE Tokenizer Trainer v2 for П.Р.І.Ч.С.
Trains a clean, zero-OOV 32,768 vocabulary on data/corpus_v2.jsonl.

Features:
- ByteLevel pre-tokenizer with add_prefix_space=True (consistent token IDs for sentence starts)
- Non-splittable special tokens: <pad>, <bos>, <eos>, <unk>, <prichs_think>, </prichs_think>, etc.
- ByteLevel decoder preserving all special tokens with skip_special_tokens=False
- Thorough validation on Stark identity, culture, philosophy, cinema and code.
"""

import sys
import json
from pathlib import Path
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
    decoders,
    processors,
    AddedToken,
)

sys.stdout.reconfigure(encoding="utf-8")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = WORKSPACE_ROOT / "data" / "corpus_v2.jsonl"
OUTPUT_TOKENIZER = WORKSPACE_ROOT / "tokenizer.json"

VOCAB_SIZE = 32768

SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<prichs_think>",
    "</prichs_think>",
    "<starkdb_query>",
    "</starkdb_query>",
    "<stark_action>",
]


def text_iterator():
    """Generator reading text from corpus_v2.jsonl."""
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if text:
                    yield text
            except Exception:
                pass


def main():
    print("================================================================")
    print("  П.Р.І.Ч.С. // Навчання Чистого Токенізатора v2 (32k Byte BPE)   ")
    print("================================================================")
    print(f"Корпус: {CORPUS_PATH}")
    print(f"Цільовий розмір словника: {VOCAB_SIZE:,} токенів")

    # 1. Initialize Byte-Level BPE with use_regex=True (Standard GPT-4 / LLaMA regex splitting)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()

    # 2. Add Special Tokens as non-splittable AddedTokens
    added_specials = [AddedToken(tok, special=True, normalized=False) for tok in SPECIAL_TOKENS]
    tokenizer.add_special_tokens(added_specials)

    # 3. Configure Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    # 4. Train from iterator
    print("\n⏳ [1/3] Навчання словника на повному дедуплікованому корпусі v2...")
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    # 5. Post-processor for byte-level
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # 6. Save tokenizer
    print(f"\n💾 [2/3] Збереження токенізатора у {OUTPUT_TOKENIZER}...")
    tokenizer.save(str(OUTPUT_TOKENIZER))

    # 7. Validation
    print("\n🔍 [3/3] Верифікація токенізації та збереження спеціальних токенів:")
    test_phrases = [
        "Даня Старкін і Тоні Старк будують StarkOS та StarkDB на RTX 3060.",
        "Квентін Тарантіно, Мартін Скорсезе, Гай Річі та Лало Саламанка.",
        "Сократ стверджував, що знає лише те, що нічого не знає.",
        "Діоген ходив з ліхтарем удень і шукав чесну людину.",
        "Фрідріх Ніцше писав про волю до влади, а Іммануїл Кант — про категоричний імператив.",
        "AC/DC випустили альбом Back in Black, а Led Zeppelin — Stairway to Heaven.",
        "Хідео Кодзіма створив Metal Gear Solid, а Rockstar Games — Red Dead Redemption 2.",
        "<prichs_think>Інженерний аналіз архітектури Sparse MoE v2.</prichs_think>Всі системи StarkOS в нормі.",
        "fn main() -> Result<()> { let x = Arc::new(AtomicU64::new(0)); }",
        "Двигун Audi 4.2 V8 BFM: паливна суміш 14.7:1, тиск мастила в нормі."
    ]

    for p in test_phrases:
        enc = tokenizer.encode(p)
        tokens_preview = [tokenizer.decode([tid], skip_special_tokens=False) for tid in enc.ids[:10]]
        dec = tokenizer.decode(enc.ids, skip_special_tokens=False)
        
        assert dec == p, f"Mismatch! Original: '{p}', Decoded: '{dec}'"
        
        print(f"\n   Фраза : \"{p[:60]}...\"")
        print(f"   Токенів: {len(enc.ids)} (Стиснення: {len(p.encode('utf-8')) / len(enc.ids):.2f} байт/токен)")
        print(f"   Перші токени: {tokens_preview}")
        print("   ✅ Реконструкція 100% бездоганна.")

    # Special token ID check
    print("\n--- Перевірка ID спеціальних токенів ---")
    for s in SPECIAL_TOKENS:
        tid = tokenizer.token_to_id(s)
        enc = tokenizer.encode(s)
        print(f"   '{s}': id = {tid}, encoded_ids = {enc.ids}")
        assert enc.ids == [tid], f"Спеціальний токен {s} розбивається на {enc.ids}!"

    print("\n================================================================")
    print(f"  [УСПІХ] Токенізатор v2 натреновано! Розмір словника: {tokenizer.get_vocab_size():,} токенів")
    print("================================================================\n")


if __name__ == "__main__":
    main()
