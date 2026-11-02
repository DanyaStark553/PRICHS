"""
compile_dataset.py — Compiles data/corpus.jsonl into binary .prichsbin format
Conforms to PrichsBinHeader specification:
- Magic: b"PRICHS01"
- Header: 64 bytes
- Tokens: uint16 little-endian stream with EOS boundaries for Document-Aware Dataloader
"""

import sys
import json
import struct
import time
from pathlib import Path
from tokenizers import Tokenizer

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = WORKSPACE_ROOT / "data" / "corpus.jsonl"
TOKENIZER_PATH = WORKSPACE_ROOT / "tokenizer.json"
OUTPUT_BIN = WORKSPACE_ROOT / "data" / "train.prichsbin"

HEADER_SIZE = 64
MAGIC = b"PRICHS01"


def main():
    print("================================================================")
    print("  П.Р.І.Ч.С. // Binary Dataset Compiler (.prichsbin)           ")
    print("================================================================")

    if not CORPUS_PATH.exists():
        print(f"[ERROR] Corpus not found: {CORPUS_PATH}")
        sys.exit(1)

    if not TOKENIZER_PATH.exists():
        print(f"[ERROR] Tokenizer not found: {TOKENIZER_PATH}")
        sys.exit(1)

    print(f"Loading Tokenizer: {TOKENIZER_PATH}...")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary Size : {vocab_size:,} tokens")

    eos_token_id = tokenizer.token_to_id("<eos>")
    if eos_token_id is None:
        eos_token_id = 2
    print(f"EOS Token ID    : {eos_token_id}")

    print(f"\nProcessing Corpus: {CORPUS_PATH} -> {OUTPUT_BIN}...")
    start_time = time.time()
    total_tokens = 0
    total_docs = 0

    OUTPUT_BIN.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_BIN, "wb") as bin_f, open(CORPUS_PATH, "r", encoding="utf-8") as corpus_f:
        # 1. Write placeholder header
        bin_f.write(b"\x00" * HEADER_SIZE)

        # 2. Stream encode each document
        buf = bytearray()
        buf_size_threshold = 2 * 1024 * 1024  # 2 MB buffer

        for line_num, line in enumerate(corpus_f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if not text:
                    continue

                encoded = tokenizer.encode(text)
                ids = encoded.ids

                if not ids:
                    continue

                # Check max u16
                valid_ids = [min(tid, 65535) for tid in ids]

                # Pack into binary format: token ids + EOS
                token_bytes = struct.pack(f"<{len(valid_ids)}H", *valid_ids)
                eos_bytes = struct.pack("<H", eos_token_id)

                buf.extend(token_bytes)
                buf.extend(eos_bytes)

                total_tokens += len(valid_ids) + 1
                total_docs += 1

                if len(buf) >= buf_size_threshold:
                    bin_f.write(buf)
                    buf.clear()

                if total_docs % 5000 == 0:
                    elapsed = time.time() - start_time
                    tok_sec = total_tokens / max(elapsed, 0.001)
                    print(f"   [PROGRESS] {total_docs:,} docs | {total_tokens:,} tokens ({tok_sec:,.0f} tok/s)")

            except Exception as e:
                pass

        # Flush remaining buffer
        if buf:
            bin_f.write(buf)
            buf.clear()

        # 3. Finalize and write header at offset 0
        bin_f.flush()
        bin_f.seek(0)

        # Header layout:
        # magic: 8 bytes
        # total_tokens: 8 bytes (u64)
        # vocab_size: 4 bytes (u32)
        # eos_token_id: 2 bytes (u16)
        # reserved: 42 bytes
        header = struct.pack(
            "<8sQI H42s",
            MAGIC,
            total_tokens,
            vocab_size,
            eos_token_id,
            b"\x00" * 42
        )
        assert len(header) == HEADER_SIZE, f"Header size mismatch: {len(header)} != {HEADER_SIZE}"
        bin_f.write(header)

    elapsed = time.time() - start_time
    file_size_mb = OUTPUT_BIN.stat().st_size / (1024 * 1024)

    print("\n================================================================")
    print("  [SUCCESS] Binary Dataset Compiled Successfully!              ")
    print(f"  File Path    : {OUTPUT_BIN}")
    print(f"  Total Tokens : {total_tokens:,}")
    print(f"  Total Docs   : {total_docs:,}")
    print(f"  File Size    : {file_size_mb:.2f} MB")
    print(f"  Compilation  : {elapsed:.2f} seconds ({total_tokens / max(elapsed, 0.001):,.0f} tokens/s)")
    print("================================================================\n")


if __name__ == "__main__":
    main()
