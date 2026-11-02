//! Production-Ready Analytical Backward Kernels for H.O.M.E.R / П.Р.І.Ч.С.
//!
//! Provides zero-heap-allocation Vector-Jacobian Products (VJP) for:
//! - Fused Cross-Entropy + Softmax Backward
//! - Fused RMSNorm Backward
//! - In-place RoPE Backward (-theta rotation)
//! - SwiGLU Activation Backward (d_G, d_U from d_Y)
//! - Unbiased DeepSeek-V3 MoE Router VJP Backward

/// Fused Cross-Entropy Forward + Backward (Нуль додаткових буферів)
/// Пише градієнт dL/dLogits прямо у `d_logits` і повертає скалярний Loss мікро-батчу
pub fn fused_cross_entropy_forward_backward(
    logits: &[f32],
    targets: &[u16],
    d_logits: &mut [f32],
    vocab_size: usize,
    loss_scale: f32,
) -> f32 {
    let mask = vec![1.0f32; targets.len()];
    fused_cross_entropy_forward_backward_masked(logits, targets, &mask, d_logits, vocab_size, loss_scale)
}

/// Fused Cross-Entropy Forward + Backward з підтримкою Loss Masking (SFT / Ignore Index)
/// Якщо mask[t] <= 0.0 (промпт або падінг), то вклад у Loss = 0 і d_logits[t, :] = 0.0
pub fn fused_cross_entropy_forward_backward_masked(
    logits: &[f32],
    targets: &[u16],
    mask: &[f32],
    d_logits: &mut [f32],
    vocab_size: usize,
    loss_scale: f32,
) -> f32 {
    let seq_len = targets.len();
    assert_eq!(mask.len(), seq_len, "Mask length must match targets length");
    let mut total_loss = 0.0f32;
    let mut active_count = 0.0f32;
    for &m in mask.iter() {
        if m > 0.0 {
            active_count += 1.0;
        }
    }

    if active_count == 0.0 {
        d_logits.fill(0.0);
        return 0.0;
    }

    let scale = loss_scale / active_count;

    for t in 0..seq_len {
        let row_offset = t * vocab_size;
        let d_row = &mut d_logits[row_offset..row_offset + vocab_size];

        if mask[t] <= 0.0 {
            // Зануляємо градієнт для промпту або падінгу
            d_row.fill(0.0);
            continue;
        }

        let logit_row = &logits[row_offset..row_offset + vocab_size];
        let target_idx = targets[t] as usize;

        // 1. Чисельно стабільний максимум для LogSumExp
        let mut max_val = f32::NEG_INFINITY;
        for &v in logit_row.iter() {
            if v > max_val {
                max_val = v;
            }
        }

        // 2. Експоненти та сума
        let mut sum_exp = 0.0f32;
        for (i, &v) in logit_row.iter().enumerate() {
            let e = (v - max_val).exp();
            d_row[i] = e; // Тимчасово зберігаємо exp у буфері градієнта
            sum_exp += e;
        }

        let inv_sum = 1.0 / sum_exp.max(1e-9);
        total_loss -= logit_row[target_idx] - max_val - sum_exp.ln();

        // 3. Злитий градієнт: (prob - 1.0 для таргета) * scale
        for i in 0..vocab_size {
            let prob = d_row[i] * inv_sum;
            let indicator = if i == target_idx { 1.0 } else { 0.0 };
            d_row[i] = (prob - indicator) * scale;
        }
    }

    total_loss / active_count
}

/// Fused RMSNorm Backward (2 проходи по L1-кешу, нуль алокацій)
pub fn rmsnorm_backward(
    d_out: &[f32],
    x: &[f32],
    gamma: &[f32],
    d_x: &mut [f32],
    d_gamma: &mut [f32],
    d_model: usize,
    eps: f32,
) {
    let num_tokens = x.len() / d_model;
    let inv_d = 1.0 / (d_model as f32);

    for t in 0..num_tokens {
        let offset = t * d_model;
        let x_row = &x[offset..offset + d_model];
        let dout_row = &d_out[offset..offset + d_model];
        let dx_row = &mut d_x[offset..offset + d_model];

        let mut sum_sq = 0.0f32;
        let mut dot_dout_gamma_x = 0.0f32;

        // Прохід 1: статистика RMS та внутрішній добуток
        for i in 0..d_model {
            let xi = x_row[i];
            sum_sq += xi * xi;
            dot_dout_gamma_x += dout_row[i] * gamma[i] * xi;
        }

        let rms_inv = 1.0 / (sum_sq * inv_d + eps).sqrt();
        let coeff = (dot_dout_gamma_x * inv_d) * (rms_inv * rms_inv);

        // Прохід 2: фінальні градієнти d_x та акумуляція d_gamma
        for i in 0..d_model {
            let xi = x_row[i];
            let di = dout_row[i];
            d_gamma[i] += di * xi * rms_inv;
            dx_row[i] += gamma[i] * rms_inv * (di - xi * coeff); // += для підтримки Residual з'єднань
        }
    }
}

/// Precompute Rotary Position Embedding cos and sin lookup tables
pub fn precompute_rope_cache(
    max_seq_len: usize,
    head_dim: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    let mut cos_cache = vec![0.0f32; max_seq_len * half_dim];
    let mut sin_cache = vec![0.0f32; max_seq_len * half_dim];

    for i in 0..half_dim {
        let exponent = (2 * i) as f32 / head_dim as f32;
        let freq = 1.0f32 / theta.powf(exponent);
        for t in 0..max_seq_len {
            let angle = t as f32 * freq;
            let offset = t * half_dim + i;
            cos_cache[offset] = angle.cos();
            sin_cache[offset] = angle.sin();
        }
    }

    (cos_cache, sin_cache)
}

/// RoPE Forward: Обертання векторів Q або K на кут +theta (in-place)
pub fn rope_forward_in_place(
    tensor: &mut [f32], // [seq_len, n_heads, head_dim]
    cos_cache: &[f32],  // [max_seq_len, head_dim / 2]
    sin_cache: &[f32],  // [max_seq_len, head_dim / 2]
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    pos_offset: usize,
) {
    let half_dim = head_dim / 2;
    for t in 0..seq_len {
        let pos = t + pos_offset;
        let trig_offset = pos * half_dim;
        for h in 0..n_heads {
            let head_offset = (t * n_heads + h) * head_dim;
            for i in 0..half_dim {
                let c = cos_cache[trig_offset + i];
                let s = sin_cache[trig_offset + i];

                let q0 = tensor[head_offset + 2 * i];
                let q1 = tensor[head_offset + 2 * i + 1];

                // R(theta) * [q0, q1]^T
                tensor[head_offset + 2 * i] = q0 * c - q1 * s;
                tensor[head_offset + 2 * i + 1] = q0 * s + q1 * c;
            }
        }
    }
}

/// RoPE Backward: Ортогональне обертання градієнтів на кут -theta
pub fn rope_backward_in_place(
    d_tensor: &mut [f32], // [seq_len, n_heads, head_dim]
    cos_cache: &[f32],    // [max_seq_len, head_dim / 2]
    sin_cache: &[f32],    // [max_seq_len, head_dim / 2]
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    pos_offset: usize,
) {
    let half_dim = head_dim / 2;
    for t in 0..seq_len {
        let pos = t + pos_offset;
        let trig_offset = pos * half_dim;
        for h in 0..n_heads {
            let head_offset = (t * n_heads + h) * head_dim;
            for i in 0..half_dim {
                let c = cos_cache[trig_offset + i];
                let s = sin_cache[trig_offset + i]; // Для обертання на -theta змінюємо знак при s

                let dq0 = d_tensor[head_offset + 2 * i];
                let dq1 = d_tensor[head_offset + 2 * i + 1];

                // R(-theta) * [dq0, dq1]^T
                d_tensor[head_offset + 2 * i] = dq0 * c + dq1 * s;
                d_tensor[head_offset + 2 * i + 1] = -dq0 * s + dq1 * c;
            }
        }
    }
}

/// Causal Softmax Backward: d_scores[i, j] = P[i, j] * (d_attn[i, j] - sum_k(d_attn[i, k]*P[i, k])) * scale
pub fn causal_softmax_backward(
    d_attn: &[f32],
    attn_probs: &[f32],
    d_scores: &mut [f32],
    seq_len: usize,
    scale: f32,
) {
    for i in 0..seq_len {
        let d_row = &d_attn[i * seq_len..i * seq_len + (i + 1)];
        let p_row = &attn_probs[i * seq_len..i * seq_len + (i + 1)];
        let out_row = &mut d_scores[i * seq_len..i * seq_len + (i + 1)];

        let mut sum_dp = 0.0f32;
        for j in 0..=i {
            sum_dp += d_row[j] * p_row[j];
        }

        for j in 0..=i {
            out_row[j] = p_row[j] * (d_row[j] - sum_dp) * scale;
        }
    }
}

/// Злитий поелементний градієнт активації SwiGLU: рахує d_G та d_U з проміжного d_Y
#[inline]
pub fn swiglu_activation_backward(
    d_y: &[f32],
    gate_pre_act: &[f32], // Матриця G = X * W_gate
    up_proj: &[f32],      // Матриця U = X * W_up
    d_gate: &mut [f32],
    d_up: &mut [f32],
) {
    for i in 0..d_y.len() {
        let g = gate_pre_act[i];
        let u = up_proj[i];
        let dy = d_y[i];

        let sig_g = 1.0 / (1.0 + (-g).exp());
        let silu_g = g * sig_g;
        let d_silu_g = sig_g * (1.0 + g * (1.0 - sig_g));

        d_up[i] = dy * silu_g;
        d_gate[i] = dy * u * d_silu_g;
    }
}

/// Unbiased DeepSeek-V3 Router VJP Backward
pub fn moe_router_unbiased_backward(
    d_expert_weights: &[f32],   // [top_k] — скалярні добутки <Expert_i(x), d_out>
    gating_weights: &[f32],     // [top_k] — чисті Softmax ваги g_i (без bias)
    selected_indices: &[usize], // [top_k] — індекси обраних експертів
    d_router_logits: &mut [f32], // [num_sparse_experts] — вихідний вектор dL/ds
) {
    d_router_logits.fill(0.0);

    let mut weighted_sum = 0.0f32;
    for k in 0..selected_indices.len() {
        weighted_sum += gating_weights[k] * d_expert_weights[k];
    }

    for (k, &expert_idx) in selected_indices.iter().enumerate() {
        let g_i = gating_weights[k];
        let dl_dg = d_expert_weights[k];
        d_router_logits[expert_idx] = g_i * (dl_dg - weighted_sum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sft_loss_masking_zeroes_prompt_and_padding_gradients() {
        let vocab_size = 4;
        // 4 tokens: 0 (prompt), 1 (prompt), 2 (response target), 3 (pad)
        let seq_len = 4;
        let logits = vec![
            1.0, 2.0, 0.5, 0.1, // t=0 (prompt)
            0.2, 0.3, 1.5, 0.8, // t=1 (prompt)
            2.0, 0.1, 0.4, 0.2, // t=2 (response target!)
            0.5, 0.5, 0.5, 0.5, // t=3 (pad)
        ];
        let targets = vec![1, 2, 0, 2];
        let mask = vec![0.0, 0.0, 1.0, 0.0]; // Only t=2 is active!
        let mut d_logits = vec![999.0f32; seq_len * vocab_size];

        let loss = fused_cross_entropy_forward_backward_masked(
            &logits,
            &targets,
            &mask,
            &mut d_logits,
            vocab_size,
            1.0,
        );

        // Verify loss is positive and finite
        assert!(loss > 0.0 && loss.is_finite());

        // Verify prompt positions (t=0, t=1) have ZERO gradients
        for i in 0..vocab_size {
            assert_eq!(d_logits[0 * vocab_size + i], 0.0, "Prompt t=0 must have zero grad");
            assert_eq!(d_logits[1 * vocab_size + i], 0.0, "Prompt t=1 must have zero grad");
        }

        // Verify padding position (t=3) has ZERO gradients
        for i in 0..vocab_size {
            assert_eq!(d_logits[3 * vocab_size + i], 0.0, "Padding t=3 must have zero grad");
        }

        // Verify active position (t=2) has NON-ZERO gradients
        let active_grads = &d_logits[2 * vocab_size..3 * vocab_size];
        let grad_sum: f32 = active_grads.iter().sum();
        // Sum of softmax cross-entropy gradients across vocab must be approximately 0
        assert!((grad_sum).abs() < 1e-5, "Sum of d_logits on active token must be 0");
        assert!(active_grads[0] < 0.0, "Target class gradient should be negative");
    }
}
