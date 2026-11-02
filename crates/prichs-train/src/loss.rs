/// Standard Cross-Entropy Loss for next-token prediction
pub fn cross_entropy_loss(logits: &[f32], target_token: usize, _vocab_size: usize) -> f32 {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let log_sum_exp = max_logit + sum_exp.ln();

    log_sum_exp - logits[target_token]
}

/// MoE Load-Balancing Loss (Prevents Expert Collapse)
///
/// L_aux = alpha * N * sum_i (f_i * P_i)
/// - f_i: fraction of tokens routed to expert i
/// - P_i: average routing probability allocated to expert i
pub fn moe_load_balancing_loss(
    expert_counts: &[usize],
    expert_prob_sums: &[f32],
    total_tokens: usize,
    alpha: f32,
) -> f32 {
    let num_experts = expert_counts.len();
    if total_tokens == 0 || num_experts == 0 {
        return 0.0;
    }

    let n = num_experts as f32;
    let mut sum_fp = 0.0f32;

    for i in 0..num_experts {
        let f_i = expert_counts[i] as f32 / total_tokens as f32;
        let p_i = expert_prob_sums[i] / total_tokens as f32;
        sum_fp += f_i * p_i;
    }

    alpha * n * sum_fp
}
