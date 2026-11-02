# 🌌 PRICHS (П.Р.І.Ч.С.)

> **Fine-Grained Sparse Mixture-of-Experts (MoE) LLM Architecture & Native Engine from Scratch in Rust & CUDA**  
> *1.54B Total Parameters / 435M Active Parameters per Token • Zero PyTorch Runtime Dependency • Symbiotic Zero-Copy Memory Bridge with StarkDB*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Language: Rust](https://img.shields.io/badge/Language-Rust-orange.svg)](https://www.rust-lang.org/)
[![CUDA: 12.x](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Ecosystem: StarkOS](https://img.shields.io/badge/Ecosystem-StarkOS-purple.svg)](https://github.com/DanyaStark553/starkdb)

---

## ⚡ What is PRICHS?

**PRICHS** (*"Просто Розумний Інтелект Чекає Свого"*) is a high-performance, fine-grained sparse Mixture-of-Experts (MoE) neural language model and standalone training/inference engine written completely from the metal up in **Rust** and **hand-optimized CUDA C++**.

Unlike conventional machine learning workflows wrapped around Python and heavy frameworks like PyTorch or vLLM, PRICHS delivers:
1. **Zero Runtime Python/PyTorch Overhead:** The model architecture, autograd graph, optimizer, data loader, tokenizer, and CUDA memory management are built natively in Rust.
2. **DeepSeekMoE-Style Fine-Grained Routing:** 1 always-active Shared Expert (voice anchor) + 8 Routed Experts (Top-2 routing per token).
3. **Auxiliary-Loss-Free Dynamic Load Balancing:** Expert dispatch bias is updated dynamically during training via moving averages ($b_i \leftarrow b_i + \gamma (\text{target} - \text{actual})$), eliminating gradient-polluting auxiliary penalties.
4. **Zero-Copy Host-Registered Memory Bridge:** Directly reads active associative graph synapses from [StarkDB](https://github.com/DanyaStark553/starkdb) across PCIe via `cudaHostRegister(CUDA_HOST_REGISTER_MAPPED)` and SeqLock in `< 0.8 µs` without CPU intervention.

---

## 🧮 Model Specification (1.54B / 435M Active)

| Parameter | Value | Engineering Rationale |
| :--- | :--- | :--- |
| **Total Parameters** | **~1.54 Billion** | Full capacity and broad world knowledge (including 49k vocabulary) |
| **Active Parameters** | **~435 Million** | 1 Shared + 2 Routed experts per token for ultra-fast consumer GPU inference |
| **Layers (Blocks)** | **24 Layers** | `PrichsBlock` (RMSNorm + GQA Attention + Residual + RMSNorm + MoE) |
| **Hidden Dimension ($d_{\text{model}}$)** | **1536** | 12 Query heads ($d_{\text{head}} = 128$) / 3 KV heads (**GQA 4:1**) |
| **Intermediate Dimension ($d_{\text{ffn}}$)** | **1280** | Granular specialized experts (DeepSeekMoE style) |
| **Expert Topology** | **1 Shared + 8 Routed** | Always-active anchor for reasoning & syntax + targeted domain specialists |
| **Context Extension** | **32,768 Tokens** | Two-stage training with ABF (Adjusted Base Frequency) RoPE ($\theta = 500{,}000$) |
| **Optimizer** | **Fused AdamW (float4)** | Custom CUDA kernel delivering **300+ GB/s VRAM throughput** on consumer GPUs |
| **Memory Bridge** | **Mysterium Zero-Copy** | PCI-e mapped host buffer sync with StarkDB associative graph memory |

---

## 🏛️ Architecture

```
                    ┌───────────────────────────────┐
                    │       Input Token Stream      │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │      RMSNorm (Native CUDA)    │
                    └───────────────┬───────────────┘
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │   Grouped Query Attention (GQA 4:1) with RoPE (32k)   │
       │   12 Q Heads (d=128)  │  3 KV Heads (d=128)           │
       └────────────────────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │        Residual Add           │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │      RMSNorm (Native CUDA)    │
                    └───────────────┬───────────────┘
                                    ▼
            ┌──────────────────────────────────────────────┐
            │   Fine-Grained Sparse Mixture-of-Experts     │
            │                                              │
            │  ┌────────────────┐      ┌────────────────┐  │
            │  │ Shared Expert  │      │ MoE Router     │  │
            │  │ (Always Active)│      │ (Aux-Loss-Free)│  │
            │  └───────┬────────┘      └───────┬────────┘  │
            │          │                       ▼           │
            │          │             Top-2 Selected Experts│
            │          │             [E0, E1, ..., E7]     │
            │          │                       │           │
            │          └───────────────┬───────┘           │
            │                          ▼                   │
            │                  SwiGLU Activation           │
            └──────────────────────────┬───────────────────┘
                                       ▼
                    ┌───────────────────────────────┐
                    │        Residual Add           │
                    └───────────────┬───────────────┘
                                    ▲
                                    │ Zero-Copy PCIe (< 0.8 µs)
                    ┌───────────────┴───────────────┐
                    │ StarkDB Graph Synapses Bridge │
                    └───────────────────────────────┘
```

---

## 📦 Workspace Crates

The codebase is organized as an atomic Cargo workspace:

* **[`crates/prichs-core`](crates/prichs-core):** Fundamental tensor types, shapes, forward/backward computational graph primitives, and CPU/GPU memory representations (`f32`, `f16`).
* **[`crates/prichs-cuda`](crates/prichs-cuda):** 1000+ lines of native CUDA C++ kernels (`src/kernels.cu`), cuBLAS wrappers, custom `build.rs` automated `nvcc` build pipeline, and POSIX / Windows Host Memory bridge (`bridge.rs`).
* **[`crates/prichs-model`](crates/prichs-model):** Full MoE architecture, RMS normalization, Grouped Query Attention, SwiGLU expert feed-forward networks, and dynamic router bias tracking.
* **[`crates/prichs-tokenizer`](crates/prichs-tokenizer):** High-speed Byte-Pair Encoding (BPE) runtime handling the 49k vocabulary, custom special tokens, and byte-level fallback mappings.
* **[`crates/prichs-train`](crates/prichs-train):** Pre-training & fine-tuning engine, cosine learning rate schedules with linear warmup, unbiased cross-entropy loss, and mmap binary data streaming.
* **[`crates/prichs-cli`](crates/prichs-cli):** Interactive terminal chat, parity test suites, blind benchmarks, and high-throughput inference server binaries.

---

## 🚀 Building & Getting Started

### Prerequisites
* **Rust:** 1.75+ (`cargo`, `rustc`)
* **NVIDIA CUDA Toolkit:** 12.0 or newer (`nvcc` in `PATH`)
* **C++ Compiler:** MSVC (Visual Studio 2019/2022) on Windows, or GCC/Clang on Linux
* **NVIDIA GPU:** Compute Capability 7.5+ (RTX 2060+, RTX 3060+, RTX 4090, A100, H100)

### 1. Build the Entire Workspace

```bash
# Clone the repository
git clone https://github.com/DanyaStark553/PRICHS.git
cd PRICHS

# Compile release binaries (nvcc will compile CUDA kernels automatically via build.rs)
cargo build --release
```

### 2. Verify with Diagnostic Tests

Run the model architecture sanity checks:

```bash
cargo test --release
```

---

## 🛠️ Data Pipeline & Tokenization

PRICHS streams training tokens directly from memory-mapped binary shards (`.prichsbin` / `u16` token IDs) to maximize PCIe throughput and eliminate Python data-loader bottlenecks.

### Pack Text into Binary Shards

```bash
# Pack custom text documents into high-speed binary shards
python tools/compile_dataset.py --input raw_data.txt --output data/shard_01.prichsbin

# Verify shard integrity and boundary tokens
python validate_shard.py data/shard_01.prichsbin
```

### Train Custom Tokenizer

```bash
python tools/train_tokenizer.py --corpus dataset.txt --vocab-size 49152 --output tokenizer.json
```

---

## 🧠 Dual-Brain Symbiosis with StarkDB

PRICHS is designed to run symbiotically alongside [StarkDB](https://github.com/DanyaStark553/starkdb), a zero-copy associative graph memory engine.

Instead of traditional slow RAG (vector search $\rightarrow$ text formatting $\rightarrow$ prompt re-tokenization):
1. **StarkDB** calculates spreading activation over concept nodes.
2. The top active synapse embeddings are written to pinned host memory.
3. PRICHS reads the synapse buffer directly from GPU VRAM via **PCIe Zero-Copy Host Mapping** in `< 0.8 µs`.
4. Synapse vectors act as dynamic virtual KV tokens or router bias offsets in the first Attention and MoE layers.

---

## 📜 License

Licensed under the [MIT License](LICENSE).  
Copyright (c) 2026 **Danya Stark**.
