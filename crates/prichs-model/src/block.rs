use anyhow::Result;
use prichs_core::ops::{add_tensors, rms_norm};
use prichs_core::Tensor;

use crate::attention::SelfAttention;
use crate::config::PrichsConfig;
use crate::moe::MoELayer;

/// Full Prichs Transformer Layer Block
///
/// Implements:
/// Residual(x + Attention(RMSNorm(x)))
/// Residual(x + MoE(RMSNorm(x), memory_bias))
#[derive(Debug, Clone)]
pub struct PrichsBlock {
    pub input_norm: Tensor,       // [hidden_dim]
    pub attention: SelfAttention,
    pub post_attn_norm: Tensor,   // [hidden_dim]
    pub moe: MoELayer,
    pub eps: f32,
}

impl PrichsBlock {
    pub fn new(config: &PrichsConfig) -> Result<Self> {
        let input_norm = Tensor::randn([config.hidden_dim], 1.0, 0.02)?;
        let post_attn_norm = Tensor::randn([config.hidden_dim], 1.0, 0.02)?;

        let attention = SelfAttention::new(
            config.hidden_dim,
            config.num_attention_heads,
            config.num_kv_heads,
        )?;

        let moe = MoELayer::new(
            config.hidden_dim,
            config.intermediate_dim,
            config.num_experts,
            config.num_experts_per_tok,
        )?;

        Ok(Self {
            input_norm,
            attention,
            post_attn_norm,
            moe,
            eps: config.rms_norm_eps,
        })
    }

    pub fn forward_with_rope(
        &self,
        x: &Tensor,
        cos_cache: &[f32],
        sin_cache: &[f32],
        pos_offset: usize,
        starkdb_concept_bias: Option<&[f32]>,
    ) -> Result<Tensor> {
        // 1. Attention Sub-layer with Pre-RMSNorm and RoPE
        let normed_x1 = rms_norm(x, &self.input_norm, self.eps)?;
        let (attn_out, _) = self.attention.forward_with_rope(&normed_x1, cos_cache, sin_cache, pos_offset)?;
        let x_residual = add_tensors(x, &attn_out)?;

        // 2. MoE Sub-layer with Pre-RMSNorm and Concept Bias
        let normed_x2 = rms_norm(&x_residual, &self.post_attn_norm, self.eps)?;
        let moe_out = self.moe.forward_with_memory(&normed_x2, starkdb_concept_bias)?;

        add_tensors(&x_residual, &moe_out)
    }

    pub fn forward(
        &self,
        x: &Tensor,
        starkdb_concept_bias: Option<&[f32]>,
    ) -> Result<Tensor> {
        let seq_len = x.dims()[0];
        let (cos_cache, sin_cache) = prichs_core::backward::precompute_rope_cache(
            seq_len.max(1024),
            self.attention.head_dim,
            10000.0,
        );
        self.forward_with_rope(x, &cos_cache, &sin_cache, 0, starkdb_concept_bias)
    }
}
