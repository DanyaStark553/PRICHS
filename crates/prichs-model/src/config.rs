use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrichsConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,           // Latent hidden dimension
    pub intermediate_dim: usize,     // Fine-Grained Expert SwiGLU dimension
    pub num_layers: usize,           // Number of PrichsBlocks
    pub num_attention_heads: usize,  // Q heads
    pub num_kv_heads: usize,         // KV heads (GQA 4:1)
    pub max_seq_len: usize,          // Context window
    pub rms_norm_eps: f32,
    pub rope_theta: f32,             // Base frequency for RoPE

    // Fine-Grained MoE Architecture
    pub num_experts: usize,          // Routed sparse experts
    pub num_shared_experts: usize,   // Shared expert (constant identity anchor)
    pub num_experts_per_tok: usize,  // Top-K routed per token (Total active: 1 Shared + 2 Routed)

    // Memory-Augmented Concept Injection
    pub memory_concept_dim: usize,
    pub num_virtual_memory_tokens: usize,
}

impl PrichsConfig {
    /// Baby-П.Р.І.Ч.С. for overnight training on RTX 3060 12GB (~52M Total / ~19M Active)
    ///
    /// Preserves full MoE topology (1 Shared + 8 Routed Top-2) and GQA (4:1)
    /// Runs at ~35,000 - 45,000 tok/s within ~3.8 GB VRAM.
    pub fn baby() -> Self {
        Self {
            vocab_size: 32768,              // 32k Byte-Level BPE Vocabulary
            hidden_dim: 512,                // 8 Q heads * 64 head_dim
            intermediate_dim: 384,          // Fine-Grained SwiGLU
            num_layers: 8,                  // 8 PrichsBlocks
            num_attention_heads: 8,         // 8 Q heads
            num_kv_heads: 2,                // 2 KV heads (GQA 4:1)
            max_seq_len: 1024,              // Micro-batch seq_len
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            num_experts: 8,                 // 8 routed sparse experts
            num_shared_experts: 1,          // 1 shared expert
            num_experts_per_tok: 2,         // Top-2 active
            memory_concept_dim: 512,
            num_virtual_memory_tokens: 16,
        }
    }

    /// Mini debug preset for quick CPU tests
    pub fn mini() -> Self {
        Self {
            vocab_size: 4096,
            hidden_dim: 256,
            intermediate_dim: 256,
            num_layers: 4,
            num_attention_heads: 4,
            num_kv_heads: 1,
            max_seq_len: 512,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            num_experts: 4,
            num_shared_experts: 1,
            num_experts_per_tok: 2,
            memory_concept_dim: 256,
            num_virtual_memory_tokens: 8,
        }
    }

    /// П.Р.І.Ч.С. v2 Heavy MoE Configuration (~472M Total / ~160M Active)
    /// Designed specifically for RTX 3060 12GB (~9.5 - 10.5 GB VRAM footprint under AdamW)
    pub fn v2_moe() -> Self {
        Self {
            vocab_size: 32768,              // 32k Byte-Level BPE Vocabulary
            hidden_dim: 1024,               // 16 Q heads * 64 head_dim
            intermediate_dim: 896,          // Fine-Grained SwiGLU Expert size
            num_layers: 16,                 // 16 PrichsBlocks deep hierarchy
            num_attention_heads: 16,        // 16 Q heads
            num_kv_heads: 4,                // 4 KV heads (GQA 4:1)
            max_seq_len: 1024,              // Micro-batch seq_len
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            num_experts: 8,                 // 8 routed sparse experts
            num_shared_experts: 1,          // 1 shared expert
            num_experts_per_tok: 2,         // Top-2 active
            memory_concept_dim: 1024,
            num_virtual_memory_tokens: 16,
        }
    }

    /// Full Calibrated 1.54B MoE Production Configuration (~1.54B Total / ~435M Active)
    pub fn production() -> Self {
        Self {
            vocab_size: 49152,
            hidden_dim: 1536,                // 12 Q heads * 128 head_dim
            intermediate_dim: 1280,          // Fine-Grained SwiGLU Expert size
            num_layers: 24,
            num_attention_heads: 12,
            num_kv_heads: 3,                 // GQA 4:1
            max_seq_len: 4096,               // Stage 1 context window
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            num_experts: 8,
            num_shared_experts: 1,
            num_experts_per_tok: 2,
            memory_concept_dim: 1536,
            num_virtual_memory_tokens: 64,
        }
    }
}
