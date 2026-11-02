use crate::config::PrichsConfig;

/// Активації одного шару PrichsBlock, які необхідно зберегти між Forward та Backward
pub struct LayerActivations {
    pub x_in: Vec<f32>,              // [T, d_model] — вхід перед Attention RMSNorm
    pub q_rot: Vec<f32>,             // [T, d_model] — повернуті Query
    pub k_rot: Vec<f32>,             // [T, kv_dim]  — повернуті Key
    pub v: Vec<f32>,                 // [T, kv_dim]  — Value
    pub attn_weights: Vec<f32>,      // [micro_batch_size, n_heads, seq_len, seq_len] — ізольовані матриці ваг
    pub attn_out_pre_proj: Vec<f32>, // [T, d_model] — вхід перед W_o
    pub x_mid: Vec<f32>,             // [T, d_model] — вхід перед MoE RMSNorm
    pub x_norm_moe: Vec<f32>,        // [T, d_model] — нормалізований вхід у роутер та експерти

    // Збережений стан роутера для кожного токена
    pub selected_experts: Vec<usize>, // [T * top_k]
    pub gating_weights: Vec<f32>,     // [T * top_k] — чисті Softmax ваги (без bias)

    // Активації Shared Експерта
    pub shared_gate_pre: Vec<f32>,   // [T, d_ffn]
    pub shared_up: Vec<f32>,         // [T, d_ffn]
    pub shared_swiglu_out: Vec<f32>, // [T, d_ffn] — вхід перед W_down

    // Активації обраних Top-K Sparse Експертів (зберігаємо тільки активні гілки!)
    pub sparse_gate_pre: Vec<f32>,   // [T * top_k, d_ffn]
    pub sparse_up: Vec<f32>,         // [T * top_k, d_ffn]
    pub sparse_swiglu_out: Vec<f32>, // [T * top_k, d_ffn]
    pub sparse_expert_out: Vec<f32>, // [T * top_k, d_model] — для скалярного добутку <Expert_i, d_out>
}

impl LayerActivations {
    pub fn allocate(micro_batch_size: usize, seq_len: usize, cfg: &PrichsConfig) -> Self {
        let num_tokens = micro_batch_size * seq_len;
        let d = cfg.hidden_dim;
        let kv_d = (cfg.hidden_dim / cfg.num_attention_heads) * cfg.num_kv_heads;
        let ffn = cfg.intermediate_dim;
        let top_k = cfg.num_experts_per_tok;

        // Точна ізольована розмірність ваг уваги: micro_batch_size * n_heads * seq_len * seq_len
        let attn_elements = micro_batch_size * cfg.num_attention_heads * seq_len * seq_len;

        Self {
            x_in: vec![0.0; num_tokens * d],
            q_rot: vec![0.0; num_tokens * d],
            k_rot: vec![0.0; num_tokens * kv_d],
            v: vec![0.0; num_tokens * kv_d],
            attn_weights: vec![0.0; attn_elements],
            attn_out_pre_proj: vec![0.0; num_tokens * d],
            x_mid: vec![0.0; num_tokens * d],
            x_norm_moe: vec![0.0; num_tokens * d],

            selected_experts: vec![0; num_tokens * top_k],
            gating_weights: vec![0.0; num_tokens * top_k],

            shared_gate_pre: vec![0.0; num_tokens * ffn],
            shared_up: vec![0.0; num_tokens * ffn],
            shared_swiglu_out: vec![0.0; num_tokens * ffn],

            sparse_gate_pre: vec![0.0; num_tokens * top_k * ffn],
            sparse_up: vec![0.0; num_tokens * top_k * ffn],
            sparse_swiglu_out: vec![0.0; num_tokens * top_k * ffn],
            sparse_expert_out: vec![0.0; num_tokens * top_k * d],
        }
    }
}

/// Глобальний Zero-Alloc буфер для всього циклу Forward + Backward
pub struct ActivationScratchpad {
    pub layers: Vec<LayerActivations>,
    pub x_final_pre_norm: Vec<f32>, // [T, d_model]
    pub x_final_normed: Vec<f32>,   // [T, d_model]
    pub logits: Vec<f32>,           // [T, vocab_size]

    // Перевикористовувані градієнтні шини (Ping-Pong Buffers для зворотного проходу)
    pub d_logits: Vec<f32>,         // [T, vocab_size]
    pub d_residual: Vec<f32>,       // [T, d_model] — головна магістраль градієнта від шару L до 0
    pub d_temp_model: Vec<f32>,     // [T, d_model] — робочий буфер для d_norm / d_attn
    pub d_q: Vec<f32>,              // [T, d_model]
    pub d_k: Vec<f32>,              // [T, kv_dim]
    pub d_v: Vec<f32>,              // [T, kv_dim]
    pub d_ffn_y: Vec<f32>,          // [T, d_ffn]
    pub d_ffn_gate: Vec<f32>,       // [T, d_ffn]
    pub d_ffn_up: Vec<f32>,         // [T, d_ffn]
    pub d_router_logits: Vec<f32>,  // [T, num_sparse_experts]
}

impl ActivationScratchpad {
    pub fn new(micro_batch_size: usize, seq_len: usize, cfg: &PrichsConfig) -> Self {
        let num_tokens = micro_batch_size * seq_len;
        let d = cfg.hidden_dim;
        let kv_d = (cfg.hidden_dim / cfg.num_attention_heads) * cfg.num_kv_heads;
        let ffn = cfg.intermediate_dim;

        let layers = (0..cfg.num_layers)
            .map(|_| LayerActivations::allocate(micro_batch_size, seq_len, cfg))
            .collect();

        Self {
            layers,
            x_final_pre_norm: vec![0.0; num_tokens * d],
            x_final_normed: vec![0.0; num_tokens * d],
            logits: vec![0.0; num_tokens * cfg.vocab_size],

            d_logits: vec![0.0; num_tokens * cfg.vocab_size],
            d_residual: vec![0.0; num_tokens * d],
            d_temp_model: vec![0.0; num_tokens * d],
            d_q: vec![0.0; num_tokens * d],
            d_k: vec![0.0; num_tokens * kv_d],
            d_v: vec![0.0; num_tokens * kv_d],
            d_ffn_y: vec![0.0; num_tokens * ffn],
            d_ffn_gate: vec![0.0; num_tokens * ffn],
            d_ffn_up: vec![0.0; num_tokens * ffn],
            d_router_logits: vec![0.0; num_tokens * cfg.num_experts],
        }
    }

    /// Повертає абсолютно точний обсяг виділеної пам'яті у мегабайтах (байти / 1024^2)
    pub fn memory_footprint_mb(&self) -> f32 {
        let top_level_bytes = (self.x_final_pre_norm.len()
            + self.x_final_normed.len()
            + self.logits.len()
            + self.d_logits.len()
            + self.d_residual.len()
            + self.d_temp_model.len()
            + self.d_q.len()
            + self.d_k.len()
            + self.d_v.len()
            + self.d_ffn_y.len()
            + self.d_ffn_gate.len()
            + self.d_ffn_up.len()
            + self.d_router_logits.len()) * 4;

        let layers_bytes: usize = self.layers.iter().map(|l| {
            (l.x_in.len()
                + l.q_rot.len()
                + l.k_rot.len()
                + l.v.len()
                + l.attn_weights.len()
                + l.attn_out_pre_proj.len()
                + l.x_mid.len()
                + l.x_norm_moe.len()
                + l.shared_gate_pre.len()
                + l.shared_up.len()
                + l.shared_swiglu_out.len()
                + l.sparse_gate_pre.len()
                + l.sparse_up.len()
                + l.sparse_swiglu_out.len()
                + l.sparse_expert_out.len()
                + l.gating_weights.len()) * 4
                + l.selected_experts.len() * std::mem::size_of::<usize>()
        }).sum();

        (top_level_bytes + layers_bytes) as f32 / (1024.0 * 1024.0)
    }
}
