use anyhow::{bail, Result};
use prichs_core::backward::{precompute_rope_cache, rope_forward_in_place};
use prichs_core::ops::matmul_2d;
use prichs_core::Tensor;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct AttentionStash {
    pub q_rope: Vec<f32>,
    pub k_rope: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_probs: Vec<f32>, // [num_heads, seq_len, seq_len]
    pub attn_concat: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SelfAttention {
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub out_proj: Tensor,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl SelfAttention {
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Result<Self> {
        let head_dim = hidden_dim / num_heads;
        if hidden_dim % num_heads != 0 {
            bail!("hidden_dim {} must be divisible by num_heads {}", hidden_dim, num_heads);
        }
        if num_heads % num_kv_heads != 0 {
            bail!("num_heads {} must be divisible by num_kv_heads {}", num_heads, num_kv_heads);
        }
        let std = (2.0 / hidden_dim as f32).sqrt();

        Ok(Self {
            q_proj: Tensor::randn([hidden_dim, hidden_dim], 0.0, std)?,
            k_proj: Tensor::randn([hidden_dim, num_kv_heads * head_dim], 0.0, std)?,
            v_proj: Tensor::randn([hidden_dim, num_kv_heads * head_dim], 0.0, std)?,
            out_proj: Tensor::randn([hidden_dim, hidden_dim], 0.0, std)?,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    /// Full Forward pass with Rotary Position Embeddings (RoPE)
    pub fn forward_with_rope(
        &self,
        x: &Tensor,
        cos_cache: &[f32],
        sin_cache: &[f32],
        pos_offset: usize,
    ) -> Result<(Tensor, AttentionStash)> {
        let seq_len = x.dims()[0];
        let q = matmul_2d(x, &self.q_proj)?;
        let k = matmul_2d(x, &self.k_proj)?;
        let v = matmul_2d(x, &self.v_proj)?;

        let mut q_data = q.as_f32_slice()?.to_vec();
        let mut k_data = k.as_f32_slice()?.to_vec();
        let v_data = v.as_f32_slice()?.to_vec();

        // 1. Apply RoPE to Q and K in-place
        rope_forward_in_place(
            &mut q_data,
            cos_cache,
            sin_cache,
            seq_len,
            self.num_heads,
            self.head_dim,
            pos_offset,
        );
        rope_forward_in_place(
            &mut k_data,
            cos_cache,
            sin_cache,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
            pos_offset,
        );

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let kv_group_size = self.num_heads / self.num_kv_heads;

        // 2. Process all attention heads in parallel across CPU / Rayon
        let head_results: Vec<(Vec<f32>, Vec<f32>)> = (0..self.num_heads)
            .into_par_iter()
            .map(|h| {
                let kv_h = h / kv_group_size;
                let q_offset = h * self.head_dim;
                let kv_offset = kv_h * self.head_dim;
                let kv_stride = self.num_kv_heads * self.head_dim;

                let mut scores = vec![0.0f32; seq_len * seq_len];
                for i in 0..seq_len {
                    let q_row = &q_data[i * self.hidden_dim + q_offset..i * self.hidden_dim + q_offset + self.head_dim];
                    let score_row = &mut scores[i * seq_len..(i + 1) * seq_len];
                    for j in 0..=i {
                        let k_row = &k_data[j * kv_stride + kv_offset..j * kv_stride + kv_offset + self.head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..self.head_dim {
                            dot += q_row[d] * k_row[d];
                        }
                        score_row[j] = dot * scale;
                    }
                }

                // Causal Softmax per row
                for i in 0..seq_len {
                    let row = &mut scores[i * seq_len..i * seq_len + (i + 1)];
                    let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for val in row.iter_mut() {
                        *val = (*val - max_val).exp();
                        sum_exp += *val;
                    }
                    let inv_sum = 1.0 / sum_exp.max(1e-9);
                    for val in row.iter_mut() {
                        *val *= inv_sum;
                    }
                }

                // Weighted sum of V values
                let mut head_out = vec![0.0f32; seq_len * self.head_dim];
                for i in 0..seq_len {
                    let weights_row = &scores[i * seq_len..i * seq_len + (i + 1)];
                    let out_row = &mut head_out[i * self.head_dim..(i + 1) * self.head_dim];
                    for j in 0..=i {
                        let weight = weights_row[j];
                        let v_row = &v_data[j * kv_stride + kv_offset..j * kv_stride + kv_offset + self.head_dim];
                        for d in 0..self.head_dim {
                            out_row[d] += weight * v_row[d];
                        }
                    }
                }

                (head_out, scores)
            })
            .collect();

        let mut attn_concat_slice = vec![0.0f32; seq_len * self.hidden_dim];
        let mut all_scores = vec![0.0f32; self.num_heads * seq_len * seq_len];

        for (h, (head_out, scores)) in head_results.into_iter().enumerate() {
            let q_offset = h * self.head_dim;
            for i in 0..seq_len {
                let src = &head_out[i * self.head_dim..(i + 1) * self.head_dim];
                let dst = &mut attn_concat_slice[i * self.hidden_dim + q_offset..i * self.hidden_dim + q_offset + self.head_dim];
                dst.copy_from_slice(src);
            }
            all_scores[h * seq_len * seq_len..(h + 1) * seq_len * seq_len].copy_from_slice(&scores);
        }

        let attn_concat = Tensor::from_f32_slice(&attn_concat_slice, [seq_len, self.hidden_dim])?;
        let out = matmul_2d(&attn_concat, &self.out_proj)?;

        let stash = AttentionStash {
            q_rope: q_data,
            k_rope: k_data,
            v: v_data,
            attn_probs: all_scores,
            attn_concat: attn_concat_slice,
        };

        Ok((out, stash))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.dims()[0];
        let (cos_cache, sin_cache) = precompute_rope_cache(seq_len.max(1024), self.head_dim, 10000.0);
        let (out, _) = self.forward_with_rope(x, &cos_cache, &sin_cache, 0)?;
        Ok(out)
    }
}
