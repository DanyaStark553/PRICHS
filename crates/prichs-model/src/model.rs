use anyhow::{bail, Context, Result};
use prichs_cuda::GpuModelEngine;
use prichs_core::backward::{
    causal_softmax_backward, fused_cross_entropy_forward_backward_masked,
    moe_router_unbiased_backward, precompute_rope_cache, rmsnorm_backward,
    rope_backward_in_place, swiglu_activation_backward,
};
use prichs_core::ops::rms_norm;
use prichs_core::Tensor;
use rand::Rng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::attention::AttentionStash;
use crate::block::PrichsBlock;
use crate::config::PrichsConfig;
use crate::moe::{ExpertForwardStash, MoEStash};

/// Full Sovereign Transformer Network for П.Р.І.Ч.С.
///
/// Features:
/// - Fine-Grained Sparse MoE (1 Shared + 8 Routed Top-2)
/// - Grouped-Query Attention (GQA 4:1)
/// - Weight Tying between Input Token Embeddings and LM Head
/// - Continuous Concept Bias Injection from StarkDB
/// - Autoregressive Advanced Generation with Temperature, Top-P, and Repetition Penalty
#[derive(Debug, Clone)]
pub struct PrichsModel {
    pub config: PrichsConfig,
    pub wte: Tensor,             // [vocab_size, hidden_dim]
    pub blocks: Vec<PrichsBlock>,
    pub final_norm: Tensor,      // [hidden_dim]
}

impl PrichsModel {
    pub fn new(config: PrichsConfig) -> Result<Self> {
        let std = 0.02f32;
        let wte = Tensor::randn([config.vocab_size, config.hidden_dim], 0.0, std)?;
        let final_norm = Tensor::randn([config.hidden_dim], 1.0, std)?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(PrichsBlock::new(&config)?);
        }

        Ok(Self {
            config,
            wte,
            blocks,
            final_norm,
        })
    }

    /// Forward pass through the entire model:
    /// Takes token IDs [seq_len], returns Logits [seq_len, vocab_size]
    pub fn forward(
        &self,
        token_ids: &[u16],
        starkdb_concept_bias: Option<&[f32]>,
    ) -> Result<Tensor> {
        let (logits, _) = self.forward_train(token_ids, starkdb_concept_bias)?;
        Ok(logits)
    }

    /// Full forward pass returning both (logits, normed_hidden_states) for backpropagation
    pub fn forward_train(
        &self,
        token_ids: &[u16],
        starkdb_concept_bias: Option<&[f32]>,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            bail!("Input token_ids cannot be empty");
        }

        let hidden_dim = self.config.hidden_dim;
        let vocab_size = self.config.vocab_size;
        let wte_data = self.wte.as_f32_slice()?;

        // 1. Token Embedding Lookup: [seq_len, hidden_dim]
        let mut x_data = vec![0.0f32; seq_len * hidden_dim];
        for (t, &tok) in token_ids.iter().enumerate() {
            let tok_idx = tok as usize;
            if tok_idx >= vocab_size {
                bail!("Token ID {} exceeds vocab size {}", tok_idx, vocab_size);
            }
            let wte_row = &wte_data[tok_idx * hidden_dim..(tok_idx + 1) * hidden_dim];
            x_data[t * hidden_dim..(t + 1) * hidden_dim].copy_from_slice(wte_row);
        }

        let mut x = Tensor::from_f32_slice(&x_data, [seq_len, hidden_dim])?;

        // 2. Transformer Blocks (Attention + MoE + Residuals)
        for block in &self.blocks {
            x = block.forward(&x, starkdb_concept_bias)?;
        }

        // 3. Final RMSNorm
        let normed_x = rms_norm(&x, &self.final_norm, self.config.rms_norm_eps)?;

        // 4. LM Head Projection (Weight Tying with WTE):
        // logits = normed_x * WTE^T -> [seq_len, vocab_size]
        let normed_data = normed_x.as_f32_slice()?;
        let mut logits_data = vec![0.0f32; seq_len * vocab_size];

        prichs_core::ops::matmul_a_bt(
            normed_data,
            wte_data,
            &mut logits_data,
            seq_len,
            hidden_dim,
            vocab_size,
        )?;

        let logits = Tensor::from_f32_slice(&logits_data, [seq_len, vocab_size])?;
        Ok((logits, normed_x))
    }

    /// Forward pass computing LM head projection ONLY for the last token position
    pub fn forward_last_token(
        &self,
        token_ids: &[u16],
        starkdb_concept_bias: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            bail!("Input token_ids cannot be empty");
        }

        let hidden_dim = self.config.hidden_dim;
        let vocab_size = self.config.vocab_size;
        let wte_data = self.wte.as_f32_slice()?;

        // 1. Token Embedding Lookup: [seq_len, hidden_dim]
        let mut x_data = vec![0.0f32; seq_len * hidden_dim];
        for (t, &tok) in token_ids.iter().enumerate() {
            let tok_idx = tok as usize;
            if tok_idx >= vocab_size {
                bail!("Token ID {} exceeds vocab size {}", tok_idx, vocab_size);
            }
            let wte_row = &wte_data[tok_idx * hidden_dim..(tok_idx + 1) * hidden_dim];
            x_data[t * hidden_dim..(t + 1) * hidden_dim].copy_from_slice(wte_row);
        }

        let mut x = Tensor::from_f32_slice(&x_data, [seq_len, hidden_dim])?;

        // 2. Transformer Blocks (Attention + MoE + Residuals)
        for block in &self.blocks {
            x = block.forward(&x, starkdb_concept_bias)?;
        }

        // 3. Final RMSNorm
        let normed_x = rms_norm(&x, &self.final_norm, self.config.rms_norm_eps)?;

        // 4. LM Head Projection ONLY for the last token position:
        let normed_data = normed_x.as_f32_slice()?;
        let last_x_row = &normed_data[(seq_len - 1) * hidden_dim..seq_len * hidden_dim];
        let mut last_logits = vec![0.0f32; vocab_size];

        prichs_core::ops::matmul_a_bt(
            last_x_row,
            wte_data,
            &mut last_logits,
            1,
            hidden_dim,
            vocab_size,
        )?;

        Ok(last_logits)
    }

    /// Advanced Token Sampler with Temperature, Top-P (Nucleus), and Repetition Penalty
    pub fn generate_advanced(
        &self,
        prompt_tokens: &[u16],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        repetition_penalty: f32,
        eos_token_id: u16,
    ) -> Result<Vec<u16>> {
        let mut tokens = prompt_tokens.to_vec();
        let mut rng = rand::thread_rng();
        let vocab_size = self.config.vocab_size;

        for _ in 0..max_new_tokens {
            let context_len = tokens.len().min(self.config.max_seq_len);
            let context_window = &tokens[tokens.len() - context_len..];

            let mut last_logits = self.forward_last_token(context_window, None)?;

            // 1. Apply Repetition Penalty on the last 64 tokens
            if repetition_penalty > 1.0 {
                let window_start = tokens.len().saturating_sub(64);
                let recent_tokens = &tokens[window_start..];
                for &t in recent_tokens {
                    let tid = t as usize;
                    if tid < vocab_size {
                        let val = last_logits[tid];
                        if val > 0.0 {
                            last_logits[tid] = val / repetition_penalty;
                        } else {
                            last_logits[tid] = val * repetition_penalty;
                        }
                    }
                }
            }

            // 2. Greedy or Temperature scaling
            let next_token: u16 = if temperature <= 0.01 {
                let mut max_idx = 0usize;
                let mut max_val = f32::NEG_INFINITY;
                for (idx, &val) in last_logits.iter().enumerate() {
                    if val > max_val {
                        max_val = val;
                        max_idx = idx;
                    }
                }
                max_idx as u16
            } else {
                let inv_temp = 1.0 / temperature;
                let max_val = last_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                let mut probs = vec![0.0f32; vocab_size];

                for i in 0..vocab_size {
                    let e = ((last_logits[i] - max_val) * inv_temp).exp();
                    probs[i] = e;
                    sum_exp += e;
                }

                let inv_sum = 1.0 / (sum_exp + 1e-9);
                for p in probs.iter_mut() {
                    *p *= inv_sum;
                }

                // 3. Top-P (Nucleus Sampling)
                if top_p < 1.0 {
                    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
                    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                    let mut cum_prob = 0.0f32;
                    let mut cutoff_idx = indexed.len();
                    for (i, &(_, p)) in indexed.iter().enumerate() {
                        cum_prob += p;
                        if cum_prob >= top_p {
                            cutoff_idx = i + 1;
                            break;
                        }
                    }

                    let filtered = &indexed[..cutoff_idx];
                    let filtered_sum: f32 = filtered.iter().map(|(_, p)| p).sum();
                    let inv_filt_sum = 1.0 / (filtered_sum + 1e-9);

                    let sample_val: f32 = rng.gen();
                    let mut acc = 0.0f32;
                    let mut chosen = filtered[0].0 as u16;

                    for &(idx, p) in filtered {
                        acc += p * inv_filt_sum;
                        if sample_val <= acc {
                            chosen = idx as u16;
                            break;
                        }
                    }
                    chosen
                } else {
                    let sample_val: f32 = rng.gen();
                    let mut cum_sum = 0.0f32;
                    let mut chosen = (vocab_size - 1) as u16;

                    for (idx, &prob) in probs.iter().enumerate() {
                        cum_sum += prob;
                        if sample_val <= cum_sum {
                            chosen = idx as u16;
                            break;
                        }
                    }
                    chosen
                }
            };

            tokens.push(next_token);
            if next_token == eos_token_id || next_token == 2 {
                break;
            }
            let cur_len = tokens.len();
            if cur_len >= 2 && tokens[cur_len - 2] == 2375 && tokens[cur_len - 1] == 34 {
                tokens.pop();
                tokens.pop();
                break;
            }
        }

        Ok(tokens)
    }

    /// Basic generate wrapper
    pub fn generate(
        &self,
        prompt_tokens: &[u16],
        max_new_tokens: usize,
        temperature: f32,
        eos_token_id: u16,
    ) -> Result<Vec<u16>> {
        self.generate_advanced(prompt_tokens, max_new_tokens, temperature, 0.9, 1.15, eos_token_id)
    }

    /// Returns all model parameter tensors mutably along with weight-decay flags
    pub fn get_parameter_tensors_mut(&mut self) -> Vec<(&mut Tensor, bool)> {
        let mut params = Vec::with_capacity(300);
        // Index 0: WTE (weight decay = true)
        params.push((&mut self.wte, true));
        // Index 1: final_norm (weight decay = false)
        params.push((&mut self.final_norm, false));

        for block in &mut self.blocks {
            params.push((&mut block.input_norm, false));
            params.push((&mut block.attention.q_proj, true));
            params.push((&mut block.attention.k_proj, true));
            params.push((&mut block.attention.v_proj, true));
            params.push((&mut block.attention.out_proj, true));
            params.push((&mut block.post_attn_norm, false));
            params.push((&mut block.moe.router, false));
            params.push((&mut block.moe.shared_expert.gate_proj, true));
            params.push((&mut block.moe.shared_expert.up_proj, true));
            params.push((&mut block.moe.shared_expert.down_proj, true));
            for exp in &mut block.moe.experts {
                params.push((&mut exp.gate_proj, true));
                params.push((&mut exp.up_proj, true));
                params.push((&mut exp.down_proj, true));
            }
        }
        params
    }

    /// Returns all model parameter tensors immutably along with weight-decay flags
    pub fn get_parameter_tensors(&self) -> Vec<(&Tensor, bool)> {
        let mut params = Vec::with_capacity(300);
        params.push((&self.wte, true));
        params.push((&self.final_norm, false));

        for block in &self.blocks {
            params.push((&block.input_norm, false));
            params.push((&block.attention.q_proj, true));
            params.push((&block.attention.k_proj, true));
            params.push((&block.attention.v_proj, true));
            params.push((&block.attention.out_proj, true));
            params.push((&block.post_attn_norm, false));
            params.push((&block.moe.router, false));
            params.push((&block.moe.shared_expert.gate_proj, true));
            params.push((&block.moe.shared_expert.up_proj, true));
            params.push((&block.moe.shared_expert.down_proj, true));
            for exp in &block.moe.experts {
                params.push((&exp.gate_proj, true));
                params.push((&exp.up_proj, true));
                params.push((&exp.down_proj, true));
            }
        }
        params
    }

    /// Full forward and analytical backward training step with optional loss masking (SFT)
    pub fn train_step_masked(
        &mut self,
        inputs: &[u16],
        targets: &[u16],
        mask: &[f32],
        grads: &mut ModelGradients,
        optimizer: &mut FullModelOptimizer,
        lr: f32,
    ) -> Result<f32> {
        let loss = self.forward_backward_masked(inputs, targets, mask, grads)?;
        optimizer.step(self, &grads.grads, lr)?;
        Ok(loss)
    }

    /// Create and initialize Persistent VRAM Engine for this model (Zero PCIe Weight Copy)
    pub fn train_step(
        &mut self,
        inputs: &[u16],
        targets: &[u16],
        grads: &mut ModelGradients,
        optimizer: &mut FullModelOptimizer,
        lr: f32,
    ) -> Result<f32> {
        let mask = vec![1.0f32; targets.len()];
        self.train_step_masked(inputs, targets, &mask, grads, optimizer, lr)
    }

    /// Computes full forward and analytical backward across all layers into `grads` without optimizer step
    pub fn forward_backward(
        &mut self,
        inputs: &[u16],
        targets: &[u16],
        grads: &mut ModelGradients,
    ) -> Result<f32> {
        let mask = vec![1.0f32; targets.len()];
        self.forward_backward_masked(inputs, targets, &mask, grads)
    }

    /// Computes full forward and analytical backward across all layers into `grads` with explicit Loss Masking (SFT)
    pub fn forward_backward_masked(
        &mut self,
        inputs: &[u16],
        targets: &[u16],
        mask: &[f32],
        grads: &mut ModelGradients,
    ) -> Result<f32> {
        let seq_len = inputs.len();
        if seq_len == 0 {
            bail!("Input tokens cannot be empty");
        }
        if targets.len() != seq_len {
            bail!("Targets length ({}) must match inputs length ({})", targets.len(), seq_len);
        }
        if mask.len() != seq_len {
            bail!("Mask length ({}) must match inputs length ({})", mask.len(), seq_len);
        }
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;
        let vocab_size = self.config.vocab_size;
        let eps = self.config.rms_norm_eps;

        // Reset gradients
        grads.zero();

        // Precompute RoPE cache for this sequence length
        let (cos_cache, sin_cache) = precompute_rope_cache(
            seq_len.max(1024),
            self.blocks[0].attention.head_dim,
            self.config.rope_theta,
        );

        // 1. Forward Pass with Activation Stashing:
        let wte_data = self.wte.as_f32_slice()?;
        let mut x_data = vec![0.0f32; seq_len * hidden_dim];
        for (t, &tok) in inputs.iter().enumerate() {
            let tok_idx = tok as usize;
            if tok_idx < vocab_size {
                let row = &wte_data[tok_idx * hidden_dim..(tok_idx + 1) * hidden_dim];
                x_data[t * hidden_dim..(t + 1) * hidden_dim].copy_from_slice(row);
            }
        }

        let mut x = Tensor::from_f32_slice(&x_data, [seq_len, hidden_dim])?;

        struct LayerTrainingStash {
            x_in: Vec<f32>,
            normed1: Vec<f32>,
            attn: AttentionStash,
            x_res: Vec<f32>,
            normed2: Vec<f32>,
            moe: MoEStash,
        }

        let mut stashes = Vec::with_capacity(self.blocks.len());

        for block in &self.blocks {
            let x_in_slice = x.as_f32_slice()?.to_vec();

            // 1. Attention Pre-RMSNorm
            let normed1 = rms_norm(&x, &block.input_norm, eps)?;
            let normed1_slice = normed1.as_f32_slice()?.to_vec();

            // 2. Attention Forward with RoPE
            let (attn_out, attn_stash) = block.attention.forward_with_rope(
                &normed1,
                &cos_cache,
                &sin_cache,
                0,
            )?;
            let x_res = prichs_core::ops::add_tensors(&x, &attn_out)?;
            let x_res_slice = x_res.as_f32_slice()?.to_vec();

            // 3. MoE Pre-RMSNorm
            let normed2 = rms_norm(&x_res, &block.post_attn_norm, eps)?;
            let normed2_slice = normed2.as_f32_slice()?.to_vec();

            // 4. Batched MoE Forward (Gather-Scatter)
            let (moe_out, moe_stash) = block.moe.forward_train(&normed2, None)?;
            x = prichs_core::ops::add_tensors(&x_res, &moe_out)?;

            stashes.push(LayerTrainingStash {
                x_in: x_in_slice,
                normed1: normed1_slice,
                attn: attn_stash,
                x_res: x_res_slice,
                normed2: normed2_slice,
                moe: moe_stash,
            });
        }

        // 2. Final RMSNorm + LM Head + Fused Cross-Entropy
        let x_last_slice = x.as_f32_slice()?.to_vec();
        let normed_final = rms_norm(&x, &self.final_norm, eps)?;
        let normed_final_slice = normed_final.as_f32_slice()?;

        let mut logits_data = vec![0.0f32; seq_len * vocab_size];
        prichs_core::ops::matmul_a_bt(
            normed_final_slice,
            self.wte.as_f32_slice()?,
            &mut logits_data,
            seq_len,
            hidden_dim,
            vocab_size,
        )?;

        let mut d_logits = vec![0.0f32; seq_len * vocab_size];
        let loss = fused_cross_entropy_forward_backward_masked(
            &logits_data,
            targets,
            mask,
            &mut d_logits,
            vocab_size,
            1.0,
        );

        // 3. LM Head Backward (Tied Token Embeddings WTE):
        // Index 0: WTE
        prichs_core::ops::matmul_at_b(
            &d_logits,
            normed_final_slice,
            &mut grads.grads[0],
            seq_len,
            vocab_size,
            hidden_dim,
        )?;

        // Gradient into normed_final: d_normed = d_logits * WTE
        let mut d_normed = vec![0.0f32; seq_len * hidden_dim];
        prichs_core::ops::matmul_gpu(
            &d_logits,
            self.wte.as_f32_slice()?,
            &mut d_normed,
            seq_len,
            vocab_size,
            hidden_dim,
        )?;

        // Index 1: final_norm
        let mut d_x = vec![0.0f32; seq_len * hidden_dim];
        rmsnorm_backward(
            &d_normed,
            &x_last_slice,
            self.final_norm.as_f32_slice()?,
            &mut d_x,
            &mut grads.grads[1],
            hidden_dim,
            eps,
        );

        // 4. Backprop through each block (from L-1 down to 0)
        let num_experts = self.config.num_experts;
        let block_param_count = 10 + num_experts * 3;

        for l in (0..self.blocks.len()).rev() {
            let stash = &stashes[l];
            let base_idx = 2 + l * block_param_count;
            let block = &mut self.blocks[l];

            // -------------------------------------------------------------
            // A. MoE Backward (Shared Expert + 8 Routed Experts + Router)
            // -------------------------------------------------------------
            let d_moe = &d_x;
            let mut d_normed2 = vec![0.0f32; seq_len * hidden_dim];

            // 1. Shared Expert Backward:
            // Down proj: grads[base_idx + 9]
            prichs_core::ops::matmul_at_b(
                &stash.moe.shared_act,
                d_moe,
                &mut grads.grads[base_idx + 9],
                seq_len,
                intermediate_dim,
                hidden_dim,
            )?;
            let mut d_shared_act = vec![0.0f32; seq_len * intermediate_dim];
            prichs_core::ops::matmul_a_bt(
                d_moe,
                block.moe.shared_expert.down_proj.as_f32_slice()?,
                &mut d_shared_act,
                seq_len,
                hidden_dim,
                intermediate_dim,
            )?;

            let mut d_shared_gate = vec![0.0f32; seq_len * intermediate_dim];
            let mut d_shared_up = vec![0.0f32; seq_len * intermediate_dim];
            swiglu_activation_backward(
                &d_shared_act,
                &stash.moe.shared_gate,
                &stash.moe.shared_up,
                &mut d_shared_gate,
                &mut d_shared_up,
            );

            // Gate & Up proj: grads[base_idx + 7] and grads[base_idx + 8]
            prichs_core::ops::matmul_at_b(
                &stash.normed2,
                &d_shared_gate,
                &mut grads.grads[base_idx + 7],
                seq_len,
                hidden_dim,
                intermediate_dim,
            )?;
            prichs_core::ops::matmul_at_b(
                &stash.normed2,
                &d_shared_up,
                &mut grads.grads[base_idx + 8],
                seq_len,
                hidden_dim,
                intermediate_dim,
            )?;

            // Shared expert input gradient into d_normed2:
            let mut d_normed2_shared = vec![0.0f32; seq_len * hidden_dim];
            let mut d_normed2_tmp = vec![0.0f32; seq_len * hidden_dim];
            prichs_core::ops::matmul_a_bt(
                &d_shared_gate,
                block.moe.shared_expert.gate_proj.as_f32_slice()?,
                &mut d_normed2_shared,
                seq_len,
                intermediate_dim,
                hidden_dim,
            )?;
            prichs_core::ops::matmul_a_bt(
                &d_shared_up,
                block.moe.shared_expert.up_proj.as_f32_slice()?,
                &mut d_normed2_tmp,
                seq_len,
                intermediate_dim,
                hidden_dim,
            )?;
            for i in 0..d_normed2.len() {
                d_normed2[i] += d_normed2_shared[i] + d_normed2_tmp[i];
            }

            // 2. 8 Routed Experts Batched Backward (cuBLAS TF32)
            let mut token_expert_scalars = vec![0.0f32; seq_len * num_experts];

            for e_stash in &stash.moe.routed_stashes {
                let e = e_stash.expert_idx;
                let n_tokens = e_stash.token_indices.len();
                if n_tokens == 0 {
                    continue;
                }

                // Construct dY_e: [n_tokens, hidden_dim]
                let mut d_y_e = vec![0.0f32; n_tokens * hidden_dim];
                for (i, (&tok_idx, &weight)) in e_stash.token_indices.iter().zip(e_stash.weights.iter()).enumerate() {
                    let d_moe_row = &d_moe[tok_idx * hidden_dim..(tok_idx + 1) * hidden_dim];
                    let d_y_row = &mut d_y_e[i * hidden_dim..(i + 1) * hidden_dim];
                    let mut dot = 0.0f32;
                    let exp_out_row = &e_stash.out[i * hidden_dim..(i + 1) * hidden_dim];
                    for d in 0..hidden_dim {
                        d_y_row[d] = weight * d_moe_row[d];
                        dot += exp_out_row[d] * d_moe_row[d];
                    }
                    token_expert_scalars[tok_idx * num_experts + e] = dot;
                }

                // Down proj: base_idx + 10 + e * 3 + 2
                prichs_core::ops::matmul_at_b(
                    &e_stash.act,
                    &d_y_e,
                    &mut grads.grads[base_idx + 10 + e * 3 + 2],
                    n_tokens,
                    intermediate_dim,
                    hidden_dim,
                )?;

                let mut d_act_e = vec![0.0f32; n_tokens * intermediate_dim];
                prichs_core::ops::matmul_a_bt(
                    &d_y_e,
                    block.moe.experts[e].down_proj.as_f32_slice()?,
                    &mut d_act_e,
                    n_tokens,
                    hidden_dim,
                    intermediate_dim,
                )?;

                let mut d_gate_e = vec![0.0f32; n_tokens * intermediate_dim];
                let mut d_up_e = vec![0.0f32; n_tokens * intermediate_dim];
                swiglu_activation_backward(
                    &d_act_e,
                    &e_stash.gate,
                    &e_stash.up,
                    &mut d_gate_e,
                    &mut d_up_e,
                );

                // Gate proj: base_idx + 10 + e * 3 + 0
                prichs_core::ops::matmul_at_b(
                    &e_stash.x,
                    &d_gate_e,
                    &mut grads.grads[base_idx + 10 + e * 3 + 0],
                    n_tokens,
                    hidden_dim,
                    intermediate_dim,
                )?;

                // Up proj: base_idx + 10 + e * 3 + 1
                prichs_core::ops::matmul_at_b(
                    &e_stash.x,
                    &d_up_e,
                    &mut grads.grads[base_idx + 10 + e * 3 + 1],
                    n_tokens,
                    hidden_dim,
                    intermediate_dim,
                )?;

                // Expert input gradient dX_e -> Scatter-Add to d_normed2
                let mut d_x_e = vec![0.0f32; n_tokens * hidden_dim];
                let mut d_x_e_tmp = vec![0.0f32; n_tokens * hidden_dim];
                prichs_core::ops::matmul_a_bt(
                    &d_gate_e,
                    block.moe.experts[e].gate_proj.as_f32_slice()?,
                    &mut d_x_e,
                    n_tokens,
                    intermediate_dim,
                    hidden_dim,
                )?;
                prichs_core::ops::matmul_a_bt(
                    &d_up_e,
                    block.moe.experts[e].up_proj.as_f32_slice()?,
                    &mut d_x_e_tmp,
                    n_tokens,
                    intermediate_dim,
                    hidden_dim,
                )?;

                for (i, (&tok_idx, &weight)) in e_stash.token_indices.iter().zip(e_stash.weights.iter()).enumerate() {
                    let out_row = &mut d_normed2[tok_idx * hidden_dim..(tok_idx + 1) * hidden_dim];
                    let dx_row = &d_x_e[i * hidden_dim..(i + 1) * hidden_dim];
                    let dx_tmp_row = &d_x_e_tmp[i * hidden_dim..(i + 1) * hidden_dim];
                    for d in 0..hidden_dim {
                        out_row[d] += weight * (dx_row[d] + dx_tmp_row[d]);
                    }
                }
            }

            // 3. Router VJP Backward (base_idx + 6):
            let top_k = block.moe.top_k;
            let mut d_router_logits = vec![0.0f32; seq_len * num_experts];
            for t in 0..seq_len {
                let indices = &stash.moe.selected_experts[t * top_k..(t + 1) * top_k];
                let weights = &stash.moe.gating_weights[t * top_k..(t + 1) * top_k];
                let mut d_expert_weights = Vec::with_capacity(top_k);
                for &idx in indices {
                    d_expert_weights.push(token_expert_scalars[t * num_experts + idx]);
                }
                let mut d_row = vec![0.0f32; num_experts];
                moe_router_unbiased_backward(&d_expert_weights, weights, indices, &mut d_row);
                d_router_logits[t * num_experts..(t + 1) * num_experts].copy_from_slice(&d_row);
            }

            // Router weight gradient
            prichs_core::ops::matmul_at_b(
                &stash.normed2,
                &d_router_logits,
                &mut grads.grads[base_idx + 6],
                seq_len,
                hidden_dim,
                num_experts,
            )?;

            // Router gradient into d_normed2
            let mut d_normed2_router = vec![0.0f32; seq_len * hidden_dim];
            prichs_core::ops::matmul_a_bt(
                &d_router_logits,
                block.moe.router.as_f32_slice()?,
                &mut d_normed2_router,
                seq_len,
                num_experts,
                hidden_dim,
            )?;
            for i in 0..d_normed2.len() {
                d_normed2[i] += d_normed2_router[i];
            }

            // DeepSeek-V3 Loss-Free router bias update
            let mut expert_counts = vec![0usize; num_experts];
            for &idx in &stash.moe.selected_experts {
                if idx < num_experts {
                    expert_counts[idx] += 1;
                }
            }
            block.moe.update_aux_free_bias(&expert_counts, seq_len, 0.001);

            // Post-attn RMSNorm: base_idx + 5
            let mut d_x_res = d_x.clone();
            rmsnorm_backward(
                &d_normed2,
                &stash.x_res,
                block.post_attn_norm.as_f32_slice()?,
                &mut d_x_res,
                &mut grads.grads[base_idx + 5],
                hidden_dim,
                eps,
            );

            // -------------------------------------------------------------
            // B. Causal Attention Backward with RoPE
            // -------------------------------------------------------------
            // base_idx + 4: out_proj
            let mut d_attn_concat = vec![0.0f32; seq_len * hidden_dim];
            prichs_core::ops::matmul_at_b(
                &stash.attn.attn_concat,
                &d_x_res,
                &mut grads.grads[base_idx + 4],
                seq_len,
                hidden_dim,
                hidden_dim,
            )?;
            prichs_core::ops::matmul_a_bt(
                &d_x_res,
                block.attention.out_proj.as_f32_slice()?,
                &mut d_attn_concat,
                seq_len,
                hidden_dim,
                hidden_dim,
            )?;

            // Attention heads backward through Causal Softmax and V
            let num_heads = block.attention.num_heads;
            let num_kv_heads = block.attention.num_kv_heads;
            let head_dim = block.attention.head_dim;
            let kv_dim = num_kv_heads * head_dim;
            let kv_group_size = num_heads / num_kv_heads;
            let scale = 1.0 / (head_dim as f32).sqrt();

            let mut d_q_rope = vec![0.0f32; seq_len * hidden_dim];
            let mut d_k_rope = vec![0.0f32; seq_len * kv_dim];
            let mut d_v = vec![0.0f32; seq_len * kv_dim];

            for h in 0..num_heads {
                let kv_h = h / kv_group_size;
                let q_offset = h * head_dim;
                let kv_offset = kv_h * head_dim;
                let probs_offset = h * seq_len * seq_len;

                let attn_probs_h = &stash.attn.attn_probs[probs_offset..probs_offset + seq_len * seq_len];

                // 1. dA = dO * V^T, dV += A^T * dO
                let mut d_a_h = vec![0.0f32; seq_len * seq_len];
                for i in 0..seq_len {
                    let do_row = &d_attn_concat[i * hidden_dim + q_offset..i * hidden_dim + q_offset + head_dim];
                    for j in 0..=i {
                        let v_row = &stash.attn.v[j * kv_dim + kv_offset..j * kv_dim + kv_offset + head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += do_row[d] * v_row[d];
                        }
                        d_a_h[i * seq_len + j] = dot;

                        let p = attn_probs_h[i * seq_len + j];
                        let dv_row = &mut d_v[j * kv_dim + kv_offset..j * kv_dim + kv_offset + head_dim];
                        for d in 0..head_dim {
                            dv_row[d] += p * do_row[d];
                        }
                    }
                }

                // 2. Causal Softmax Backward
                let mut d_scores_h = vec![0.0f32; seq_len * seq_len];
                causal_softmax_backward(
                    &d_a_h,
                    attn_probs_h,
                    &mut d_scores_h,
                    seq_len,
                    scale,
                );

                // 3. dQ = dS * K, dK += dS^T * Q
                for i in 0..seq_len {
                    let dq_row = &mut d_q_rope[i * hidden_dim + q_offset..i * hidden_dim + q_offset + head_dim];
                    let q_row = &stash.attn.q_rope[i * hidden_dim + q_offset..i * hidden_dim + q_offset + head_dim];
                    for j in 0..=i {
                        let ds = d_scores_h[i * seq_len + j];
                        let k_row = &stash.attn.k_rope[j * kv_dim + kv_offset..j * kv_dim + kv_offset + head_dim];
                        let dk_row = &mut d_k_rope[j * kv_dim + kv_offset..j * kv_dim + kv_offset + head_dim];
                        for d in 0..head_dim {
                            dq_row[d] += ds * k_row[d];
                            dk_row[d] += ds * q_row[d];
                        }
                    }
                }
            }

            // 4. RoPE Backward (-theta rotation to get unrotated dQ and dK)
            rope_backward_in_place(
                &mut d_q_rope,
                &cos_cache,
                &sin_cache,
                seq_len,
                num_heads,
                head_dim,
                0,
            );
            rope_backward_in_place(
                &mut d_k_rope,
                &cos_cache,
                &sin_cache,
                seq_len,
                num_kv_heads,
                head_dim,
                0,
            );

            // Projections backward:
            // base_idx + 1: q_proj
            prichs_core::ops::matmul_at_b(
                &stash.normed1,
                &d_q_rope,
                &mut grads.grads[base_idx + 1],
                seq_len,
                hidden_dim,
                hidden_dim,
            )?;
            // base_idx + 2: k_proj
            prichs_core::ops::matmul_at_b(
                &stash.normed1,
                &d_k_rope,
                &mut grads.grads[base_idx + 2],
                seq_len,
                hidden_dim,
                kv_dim,
            )?;
            // base_idx + 3: v_proj
            prichs_core::ops::matmul_at_b(
                &stash.normed1,
                &d_v,
                &mut grads.grads[base_idx + 3],
                seq_len,
                hidden_dim,
                kv_dim,
            )?;

            // Input gradient through Q, K, V into d_normed1
            let mut d_normed1 = vec![0.0f32; seq_len * hidden_dim];
            let mut d_normed1_k = vec![0.0f32; seq_len * hidden_dim];
            let mut d_normed1_v = vec![0.0f32; seq_len * hidden_dim];
            prichs_core::ops::matmul_a_bt(
                &d_q_rope,
                block.attention.q_proj.as_f32_slice()?,
                &mut d_normed1,
                seq_len,
                hidden_dim,
                hidden_dim,
            )?;
            prichs_core::ops::matmul_a_bt(
                &d_k_rope,
                block.attention.k_proj.as_f32_slice()?,
                &mut d_normed1_k,
                seq_len,
                kv_dim,
                hidden_dim,
            )?;
            prichs_core::ops::matmul_a_bt(
                &d_v,
                block.attention.v_proj.as_f32_slice()?,
                &mut d_normed1_v,
                seq_len,
                kv_dim,
                hidden_dim,
            )?;
            for i in 0..d_normed1.len() {
                d_normed1[i] += d_normed1_k[i] + d_normed1_v[i];
            }

            // Input RMSNorm: base_idx + 0
            d_x = d_x_res;
            rmsnorm_backward(
                &d_normed1,
                &stash.x_in,
                block.input_norm.as_f32_slice()?,
                &mut d_x,
                &mut grads.grads[base_idx + 0],
                hidden_dim,
                eps,
            );
        }

        // 5. Input token embedding gradients (WTE)
        let grad_wte = &mut grads.grads[0];
        for t in 0..seq_len {
            let tok = inputs[t] as usize;
            if tok < vocab_size {
                for h in 0..hidden_dim {
                    grad_wte[tok * hidden_dim + h] += d_x[t * hidden_dim + h];
                }
            }
        }

        Ok(loss)
    }

    /// Create and initialize Persistent VRAM Engine for this model (Zero PCIe Weight Copy)
    pub fn create_gpu_engine(&self) -> Result<GpuModelEngine> {
        let params = self.get_parameter_tensors();
        let mut shapes = Vec::with_capacity(params.len());
        for (tensor, apply_wd) in &params {
            let dims = tensor.dims();
            let (rows, cols) = if dims.len() == 2 {
                (dims[0], dims[1])
            } else if dims.len() == 1 {
                (1, dims[0])
            } else {
                (dims[0], dims[1..].iter().product())
            };
            shapes.push((rows, cols, *apply_wd));
        }

        let mut engine = GpuModelEngine::new(&shapes)?;
        let host_slices: Vec<&[f32]> = params.iter().map(|(t, _)| t.as_f32_slice().unwrap()).collect();
        engine.upload_weights(&host_slices)?;
        Ok(engine)
    }

    /// Synchronize persistent VRAM weights back to CPU model (e.g. for checkpoint saving)
    pub fn sync_weights_from_engine(&mut self, engine: &GpuModelEngine) -> Result<()> {
        let mut params = self.get_parameter_tensors_mut();
        let mut host_slices: Vec<&mut [f32]> = Vec::with_capacity(params.len());
        for (tensor, _) in params.iter_mut() {
            host_slices.push(tensor.as_f32_mut_slice()?);
        }
        engine.download_weights(&mut host_slices)?;
        Ok(())
    }

    /// Full training step directly on Persistent GPU VRAM Engine (Zero PCIe Weight Copies)
    pub fn train_step_vram(
        &mut self,
        engine: &mut GpuModelEngine,
        inputs: &[u16],
        targets: &[u16],
        mask: &[f32],
        lr: f32,
        step: usize,
    ) -> Result<f32> {
        let loss = self.forward_backward_vram(engine, inputs, targets, mask)?;
        engine.adamw_step(lr, 0.9, 0.95, 1e-8, 0.1, step)?;
        Ok(loss)
    }

    /// Pure GPU VRAM Forward Pass (Inference only, zero backward, returns full logits [seq_len, vocab_size])
    pub fn forward_gpu_only(
        &mut self,
        engine: &mut GpuModelEngine,
        inputs: &[u16],
    ) -> Result<Vec<f32>> {
        let seq_len = inputs.len();
        if seq_len == 0 {
            bail!("Input tokens cannot be empty");
        }
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;
        let vocab_size = self.config.vocab_size;
        let eps = self.config.rms_norm_eps;

        let (cos_cache, sin_cache) = precompute_rope_cache(
            seq_len.max(1024),
            self.blocks[0].attention.head_dim,
            self.config.rope_theta,
        );
        engine.upload_rope_cache(&cos_cache, &sin_cache)?;

        let wte_data = self.wte.as_f32_slice()?;
        let mut x_data = vec![0.0f32; seq_len * hidden_dim];
        for (t, &tok) in inputs.iter().enumerate() {
            let tok_idx = tok as usize;
            if tok_idx < vocab_size {
                let wte_row = &wte_data[tok_idx * hidden_dim..(tok_idx + 1) * hidden_dim];
                x_data[t * hidden_dim..(t + 1) * hidden_dim].copy_from_slice(wte_row);
            }
        }
        let mut x = Tensor::from_f32_slice(&x_data, [seq_len, hidden_dim])?;

        let num_experts = self.config.num_experts;
        let block_param_count = 10 + num_experts * 3;

        for (l, block) in self.blocks.iter_mut().enumerate() {
            let base_idx = 2 + l * block_param_count;
            let x_in_slice = x.as_f32_slice()?.to_vec();

            // A. Attention Pre-RMSNorm on GPU
            let mut normed1 = vec![0.0f32; seq_len * hidden_dim];
            let mut rstd1 = vec![0.0f32; seq_len];
            engine.gpu_rmsnorm_forward(&x_in_slice, base_idx + 0, &mut normed1, &mut rstd1, seq_len, hidden_dim, eps)?;

            // B. GPU Batched Attention Forward
            let num_heads = block.attention.num_heads;
            let num_kv_heads = block.attention.num_kv_heads;
            let head_dim = block.attention.head_dim;
            let mut attn_out = vec![0.0f32; seq_len * hidden_dim];

            let _ = engine.gpu_batched_attention_forward(
                &normed1,
                base_idx + 1,
                base_idx + 2,
                base_idx + 3,
                base_idx + 4,
                &mut attn_out,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
            )?;

            let mut x_res = vec![0.0f32; seq_len * hidden_dim];
            for i in 0..x_res.len() {
                x_res[i] = x_in_slice[i] + attn_out[i];
            }

            // C. MoE Pre-RMSNorm on GPU
            let mut normed2 = vec![0.0f32; seq_len * hidden_dim];
            let mut rstd2 = vec![0.0f32; seq_len];
            engine.gpu_rmsnorm_forward(&x_res, base_idx + 5, &mut normed2, &mut rstd2, seq_len, hidden_dim, eps)?;

            // D. MoE Forward 100% in VRAM:
            let _ = engine.gpu_moe_forward_shared(
                &normed2,
                base_idx + 7,
                base_idx + 8,
                base_idx + 9,
                seq_len,
                hidden_dim,
                intermediate_dim,
            )?;

            let mut router_logits = vec![0.0f32; seq_len * num_experts];
            engine.forward_gemm(base_idx + 6, &normed2, &mut router_logits, seq_len, hidden_dim, num_experts)?;

            let top_k = block.moe.top_k;
            let mut expert_assignments: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];

            for t in 0..seq_len {
                let raw_logits = &router_logits[t * num_experts..(t + 1) * num_experts];
                let mut adjusted_logits = Vec::with_capacity(num_experts);
                for e in 0..num_experts {
                    adjusted_logits.push(raw_logits[e] + block.moe.router_bias[e]);
                }
                let gating = prichs_core::ops::top_k_gating_unbiased(raw_logits, &adjusted_logits, top_k);
                for (&exp_idx, &weight) in gating.indices.iter().zip(gating.weights.iter()) {
                    expert_assignments[exp_idx].push((t, weight));
                }
            }

            for (e, assignments) in expert_assignments.into_iter().enumerate() {
                let n_tokens = assignments.len();
                if n_tokens == 0 { continue; }

                let mut batch_token_indices = Vec::with_capacity(n_tokens);
                let mut batch_weights = Vec::with_capacity(n_tokens);
                for &(tok_idx, weight) in &assignments {
                    batch_token_indices.push(tok_idx as i32);
                    batch_weights.push(weight);
                }

                let expert_base = base_idx + 10 + e * 3;
                let _ = engine.gpu_moe_forward_expert(
                    &batch_token_indices,
                    &batch_weights,
                    expert_base + 0,
                    expert_base + 1,
                    expert_base + 2,
                    hidden_dim,
                    intermediate_dim,
                )?;
            }

            let mut total_moe_out = vec![0.0f32; seq_len * hidden_dim];
            engine.gpu_moe_finalize_forward(&mut total_moe_out)?;

            for i in 0..x_res.len() {
                x_res[i] += total_moe_out[i];
            }
            x = Tensor::from_f32_slice(&x_res, [seq_len, hidden_dim])?;
        }

        // Final RMSNorm on GPU
        let mut normed_final = vec![0.0f32; seq_len * hidden_dim];
        let mut rstd_final = vec![0.0f32; seq_len];
        engine.gpu_rmsnorm_forward(x.as_f32_slice()?, 1, &mut normed_final, &mut rstd_final, seq_len, hidden_dim, eps)?;

        // LM Head Projection on GPU via forward_transposed_gemm
        let mut logits = vec![0.0f32; seq_len * vocab_size];
        engine.forward_transposed_gemm(0, &normed_final, &mut logits, seq_len, hidden_dim, vocab_size)?;

        Ok(logits)
    }

    /// Forward and backward pass executing all GEMM operations directly on Persistent VRAM
    pub fn forward_backward_vram(
        &mut self,
        engine: &mut GpuModelEngine,
        inputs: &[u16],
        targets: &[u16],
        mask: &[f32],
    ) -> Result<f32> {
        engine.zero_grads()?;
        self.forward_backward_vram_accum(engine, inputs, targets, mask, 1.0)
    }

    /// Forward and backward pass with gradient accumulation across micro-batches (does NOT zero grads)
    pub fn forward_backward_vram_accum(
        &mut self,
        engine: &mut GpuModelEngine,
        inputs: &[u16],
        targets: &[u16],
        mask: &[f32],
        grad_scale: f32,
    ) -> Result<f32> {
        let seq_len = inputs.len();
        if seq_len == 0 {
            bail!("Input tokens cannot be empty");
        }
        if targets.len() != seq_len {
            bail!("Targets length ({}) must match inputs length ({})", targets.len(), seq_len);
        }
        if mask.len() != seq_len {
            bail!("Mask length ({}) must match inputs length ({})", mask.len(), seq_len);
        }

        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;
        let vocab_size = self.config.vocab_size;
        let eps = self.config.rms_norm_eps;

        let (cos_cache, sin_cache) = precompute_rope_cache(
            seq_len.max(1024),
            self.blocks[0].attention.head_dim,
            self.config.rope_theta,
        );
        engine.upload_rope_cache(&cos_cache, &sin_cache)?;

        let wte_data = self.wte.as_f32_slice()?;
        let mut x_data = vec![0.0f32; seq_len * hidden_dim];
        for (t, &tok) in inputs.iter().enumerate() {
            let tok_idx = tok as usize;
            if tok_idx < vocab_size {
                let wte_row = &wte_data[tok_idx * hidden_dim..(tok_idx + 1) * hidden_dim];
                x_data[t * hidden_dim..(t + 1) * hidden_dim].copy_from_slice(wte_row);
            }
        }
        let mut x = Tensor::from_f32_slice(&x_data, [seq_len, hidden_dim])?;

        struct LayerTrainingStash {
            x_in: Vec<f32>,
            normed1: Vec<f32>,
            rstd1: Vec<f32>,
            attn: AttentionStash,
            x_res: Vec<f32>,
            normed2: Vec<f32>,
            rstd2: Vec<f32>,
            moe: MoEStash,
        }

        let mut stashes = Vec::with_capacity(self.blocks.len());
        let num_experts = self.config.num_experts;
        let block_param_count = 10 + num_experts * 3;

        let mut t_attn_fwd = std::time::Duration::ZERO;
        let mut t_moe_fwd = std::time::Duration::ZERO;
        let mut t_norms = std::time::Duration::ZERO;
        let mut t_attn_bwd = std::time::Duration::ZERO;
        let mut t_moe_bwd = std::time::Duration::ZERO;
        let mut t_lm_head = std::time::Duration::ZERO;

        // 2. Forward pass over all transformer blocks
        for (l, block) in self.blocks.iter_mut().enumerate() {
            let base_idx = 2 + l * block_param_count;
            let x_in_slice = x.as_f32_slice()?.to_vec();

            // A. Attention Pre-RMSNorm on GPU
            let t_n1 = std::time::Instant::now();
            let mut normed1 = vec![0.0f32; seq_len * hidden_dim];
            let mut rstd1 = vec![0.0f32; seq_len];
            engine.gpu_rmsnorm_forward(&x_in_slice, base_idx + 0, &mut normed1, &mut rstd1, seq_len, hidden_dim, eps)?;
            t_norms += t_n1.elapsed();

            // B. GPU Batched Attention Forward (Zero Host Computation & Zero PCIe Ping-Pong)
            let t_af = std::time::Instant::now();
            let num_heads = block.attention.num_heads;
            let num_kv_heads = block.attention.num_kv_heads;
            let head_dim = block.attention.head_dim;
            let mut attn_out = vec![0.0f32; seq_len * hidden_dim];

            let (q_rope, k_rope, v_data, attn_probs, attn_concat) = engine.gpu_batched_attention_forward(
                &normed1,
                base_idx + 1,
                base_idx + 2,
                base_idx + 3,
                base_idx + 4,
                &mut attn_out,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
            )?;
            t_attn_fwd += t_af.elapsed();

            let attn_stash = AttentionStash {
                q_rope,
                k_rope,
                v: v_data,
                attn_probs,
                attn_concat,
            };

            let mut x_res = vec![0.0f32; seq_len * hidden_dim];
            for i in 0..x_res.len() {
                x_res[i] = x_in_slice[i] + attn_out[i];
            }

            // C. MoE Pre-RMSNorm on GPU
            let t_n2 = std::time::Instant::now();
            let mut normed2 = vec![0.0f32; seq_len * hidden_dim];
            let mut rstd2 = vec![0.0f32; seq_len];
            engine.gpu_rmsnorm_forward(&x_res, base_idx + 5, &mut normed2, &mut rstd2, seq_len, hidden_dim, eps)?;
            t_norms += t_n2.elapsed();

            // D. MoE Forward 100% in VRAM:
            let t_mf = std::time::Instant::now();
            // 1) Shared Expert Forward in VRAM:
            let (shared_gate, shared_up, shared_act, shared_out) = engine.gpu_moe_forward_shared(
                &normed2,
                base_idx + 7,
                base_idx + 8,
                base_idx + 9,
                seq_len,
                hidden_dim,
                intermediate_dim,
            )?;

            // 2) Router Logits in VRAM:
            let mut router_logits = vec![0.0f32; seq_len * num_experts];
            engine.forward_gemm(base_idx + 6, &normed2, &mut router_logits, seq_len, hidden_dim, num_experts)?;

            // 3) Token Top-K Gating:
            let top_k = block.moe.top_k;
            let mut selected_experts = vec![0usize; seq_len * top_k];
            let mut gating_weights = vec![0.0f32; seq_len * top_k];
            let mut expert_assignments: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];

            for t in 0..seq_len {
                let raw_logits = &router_logits[t * num_experts..(t + 1) * num_experts];
                let mut adjusted_logits = Vec::with_capacity(num_experts);
                for e in 0..num_experts {
                    adjusted_logits.push(raw_logits[e] + block.moe.router_bias[e]);
                }
                let gating = prichs_core::ops::top_k_gating_unbiased(raw_logits, &adjusted_logits, top_k);
                for (k, (&exp_idx, &weight)) in gating.indices.iter().zip(gating.weights.iter()).enumerate() {
                    selected_experts[t * top_k + k] = exp_idx;
                    gating_weights[t * top_k + k] = weight;
                    expert_assignments[exp_idx].push((t, weight));
                }
            }

            // 4) Routed Experts Forward directly on VRAM:
            let mut routed_stashes = Vec::with_capacity(num_experts);

            for (e, assignments) in expert_assignments.into_iter().enumerate() {
                let n_tokens = assignments.len();
                if n_tokens == 0 { continue; }

                let mut batch_token_indices = Vec::with_capacity(n_tokens);
                let mut batch_weights = Vec::with_capacity(n_tokens);
                for &(tok_idx, weight) in &assignments {
                    batch_token_indices.push(tok_idx as i32);
                    batch_weights.push(weight);
                }

                let expert_base = base_idx + 10 + e * 3;
                let (exp_x, exp_gate, exp_up, exp_act, exp_out) = engine.gpu_moe_forward_expert(
                    &batch_token_indices,
                    &batch_weights,
                    expert_base + 0,
                    expert_base + 1,
                    expert_base + 2,
                    hidden_dim,
                    intermediate_dim,
                )?;

                routed_stashes.push(ExpertForwardStash {
                    expert_idx: e,
                    token_indices: batch_token_indices.iter().map(|&i| i as usize).collect(),
                    weights: batch_weights,
                    x: exp_x,
                    gate: exp_gate,
                    up: exp_up,
                    act: exp_act,
                    out: exp_out,
                });
            }

            let mut total_moe_out = vec![0.0f32; seq_len * hidden_dim];
            engine.gpu_moe_finalize_forward(&mut total_moe_out)?;
            t_moe_fwd += t_mf.elapsed();

            for i in 0..x_res.len() {
                x_res[i] += total_moe_out[i];
            }
            x = Tensor::from_f32_slice(&x_res, [seq_len, hidden_dim])?;

            let moe_stash = MoEStash {
                router_logits,
                selected_experts,
                gating_weights,
                shared_gate,
                shared_up,
                shared_act,
                shared_out,
                routed_stashes,
            };

            stashes.push(LayerTrainingStash {
                x_in: x_in_slice,
                normed1,
                rstd1,
                attn: attn_stash,
                x_res,
                normed2,
                rstd2,
                moe: moe_stash,
            });
        }

        // 3. Final RMSNorm on GPU + GPU Fused Cross-Entropy (Zero Logits PCIe ping-pong!)
        let t_n3 = std::time::Instant::now();
        let mut normed_final = vec![0.0f32; seq_len * hidden_dim];
        let mut rstd_final = vec![0.0f32; seq_len];
        engine.gpu_rmsnorm_forward(x.as_f32_slice()?, 1, &mut normed_final, &mut rstd_final, seq_len, hidden_dim, eps)?;
        t_norms += t_n3.elapsed();

        let t_lm = std::time::Instant::now();
        let mut d_normed_final = vec![0.0f32; seq_len * hidden_dim];
        let loss = engine.gpu_fused_cross_entropy(
            &normed_final,
            targets,
            mask,
            0, // Param 0: WTE
            seq_len,
            hidden_dim,
            vocab_size,
            &mut d_normed_final,
            grad_scale,
        )?;
        t_lm_head += t_lm.elapsed();

        let t_n4 = std::time::Instant::now();
        let mut d_x = vec![0.0f32; seq_len * hidden_dim];
        engine.gpu_rmsnorm_backward(
            &d_normed_final,
            x.as_f32_slice()?,
            &rstd_final,
            1,
            &mut d_x,
            seq_len,
            hidden_dim,
        )?;
        t_norms += t_n4.elapsed();

        // 4. Backward through blocks (from L-1 down to 0)
        for l in (0..self.blocks.len()).rev() {
            let stash = &stashes[l];
            let base_idx = 2 + l * block_param_count;
            let block = &mut self.blocks[l];

            let d_moe = &d_x;

            // 1. Initialize MoE backward on GPU
            let t_mb = std::time::Instant::now();
            engine.gpu_moe_init_backward(d_moe)?;

            // 2. Routed Experts backward in VRAM:
            let mut token_expert_scalars = vec![0.0f32; seq_len * num_experts];
            for e_stash in &stash.moe.routed_stashes {
                let e = e_stash.expert_idx;
                let n_tokens = e_stash.token_indices.len();
                if n_tokens == 0 { continue; }

                let batch_token_indices: Vec<i32> = e_stash.token_indices.iter().map(|&i| i as i32).collect();
                let expert_base = base_idx + 10 + e * 3;

                let scalars = engine.gpu_moe_backward_expert(
                    &batch_token_indices,
                    &e_stash.weights,
                    &e_stash.x,
                    &e_stash.gate,
                    &e_stash.up,
                    &e_stash.act,
                    &e_stash.out,
                    expert_base + 0,
                    expert_base + 1,
                    expert_base + 2,
                    hidden_dim,
                    intermediate_dim,
                )?;

                for (&tok_idx, &scalar) in e_stash.token_indices.iter().zip(scalars.iter()) {
                    token_expert_scalars[tok_idx * num_experts + e] = scalar;
                }
            }

            // 3. Shared Expert backward in VRAM:
            engine.gpu_moe_backward_shared(
                &stash.moe.shared_act,
                &stash.moe.shared_gate,
                &stash.moe.shared_up,
                &stash.normed2,
                base_idx + 7,
                base_idx + 8,
                base_idx + 9,
                seq_len,
                hidden_dim,
                intermediate_dim,
            )?;

            // Synchronize GPU stream before host CPU reads token_expert_scalars
            engine.sync_stream()?;

            // 4. Router backward VJP on CPU:
            let top_k = block.moe.top_k;
            let mut d_router_logits = vec![0.0f32; seq_len * num_experts];
            for t in 0..seq_len {
                let indices = &stash.moe.selected_experts[t * top_k..(t + 1) * top_k];
                let weights = &stash.moe.gating_weights[t * top_k..(t + 1) * top_k];
                let mut d_weights = [0.0f32; 8];
                for (k, &exp_idx) in indices.iter().enumerate() {
                    d_weights[k] = token_expert_scalars[t * num_experts + exp_idx];
                }
                let d_logits_row = &mut d_router_logits[t * num_experts..(t + 1) * num_experts];
                moe_router_unbiased_backward(&d_weights[..top_k], weights, indices, d_logits_row);
            }

            // DeepSeek-V3 Loss-Free router bias update
            let mut expert_counts = vec![0usize; num_experts];
            for &idx in &stash.moe.selected_experts {
                if idx < num_experts {
                    expert_counts[idx] += 1;
                }
            }
            block.moe.update_aux_free_bias(&expert_counts, seq_len, 0.001);

            // 5. Router backward in VRAM & download accumulated d_normed2:
            let mut d_normed2 = vec![0.0f32; seq_len * hidden_dim];
            engine.gpu_moe_backward_router(
                &stash.normed2,
                &d_router_logits,
                base_idx + 6,
                seq_len,
                hidden_dim,
                num_experts,
                &mut d_normed2,
            )?;
            t_moe_bwd += t_mb.elapsed();

            // 6. Post-Attn RMSNorm backward on GPU:
            let t_n5 = std::time::Instant::now();
            let mut d_x_norm2 = vec![0.0f32; seq_len * hidden_dim];
            engine.gpu_rmsnorm_backward(
                &d_normed2,
                &stash.x_res,
                &stash.rstd2,
                base_idx + 5,
                &mut d_x_norm2,
                seq_len,
                hidden_dim,
            )?;
            t_norms += t_n5.elapsed();

            let mut d_x_res = vec![0.0f32; seq_len * hidden_dim];
            for i in 0..d_x_res.len() {
                d_x_res[i] = d_moe[i] + d_x_norm2[i];
            }

            // 7. Attention backward on GPU (Batched cuBLAS GEMMs + CUDA Softmax Backward):
            let t_ab = std::time::Instant::now();
            let num_heads = block.attention.num_heads;
            let num_kv_heads = block.attention.num_kv_heads;
            let head_dim = block.attention.head_dim;
            let mut d_normed1 = vec![0.0f32; seq_len * hidden_dim];

            engine.gpu_batched_attention_backward(
                &d_x_res,
                &stash.normed1,
                &stash.attn.q_rope,
                &stash.attn.k_rope,
                &stash.attn.v,
                &stash.attn.attn_probs,
                &stash.attn.attn_concat,
                base_idx + 1,
                base_idx + 2,
                base_idx + 3,
                base_idx + 4,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
                &mut d_normed1,
            )?;
            t_attn_bwd += t_ab.elapsed();

            // 8. Pre-Attn RMSNorm backward on GPU:
            let t_n6 = std::time::Instant::now();
            let mut d_x_norm1 = vec![0.0f32; seq_len * hidden_dim];
            engine.gpu_rmsnorm_backward(
                &d_normed1,
                &stash.x_in,
                &stash.rstd1,
                base_idx + 0,
                &mut d_x_norm1,
                seq_len,
                hidden_dim,
            )?;
            t_norms += t_n6.elapsed();

            for i in 0..d_x.len() {
                d_x[i] = d_x_res[i] + d_x_norm1[i];
            }
        }

        // 5. Input token embedding gradients (WTE)
        engine.accumulate_token_embeddings_grad(inputs, &d_x, hidden_dim)?;

        static PROFILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !PROFILED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            println!("\n🔍 [ПРОФІЛЮВАННЯ КРОКУ НАВЧАННЯ]:");
            println!("   > Attention Forward (Q/K/V + Softmax + Out):   {:.2?}", t_attn_fwd);
            println!("   > Attention Backward (Out + Causal + Q/K/V):   {:.2?}", t_attn_bwd);
            println!("   > MoE Forward (Shared + Router + Routed):      {:.2?}", t_moe_fwd);
            println!("   > MoE Backward (Experts + Router + Shared):    {:.2?}", t_moe_bwd);
            println!("   > LM Head + CrossEntropy (Logits + Loss):      {:.2?}", t_lm_head);
            println!("   > RMSNorms (Forward + Backward):               {:.2?}", t_norms);
            println!("   > СУМАРНО F+B:                                 {:.2?}\n",
                t_attn_fwd + t_attn_bwd + t_moe_fwd + t_moe_bwd + t_lm_head + t_norms);
        }

        Ok(loss)
    }

    /// Save weights to .prichs checkpoint file (Supports full model serialization)
    pub fn save_checkpoint(&self, path: &Path, step: usize, loss: f32) -> Result<()> {
        let file = File::create(path).context("Failed to create checkpoint file")?;
        let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file);

        writer.write_all(b"PRICHS02")?;
        writer.write_all(&(step as u64).to_le_bytes())?;
        writer.write_all(&loss.to_le_bytes())?;

        let params = self.get_parameter_tensors();
        writer.write_all(&(params.len() as u64).to_le_bytes())?;

        for (tensor, _) in params {
            let slice = tensor.as_f32_slice()?;
            writer.write_all(&(slice.len() as u64).to_le_bytes())?;
            let raw_bytes = unsafe {
                std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4)
            };
            writer.write_all(raw_bytes)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load weights from .prichs checkpoint file (Full V2 + Legacy PRICHS_W compatibility)
    pub fn load_checkpoint(&mut self, path: &Path) -> Result<(usize, f32)> {
        let file = File::open(path).context("Failed to open checkpoint file")?;
        let mut reader = BufReader::with_capacity(16 * 1024 * 1024, file);

        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;

        let mut step_bytes = [0u8; 8];
        reader.read_exact(&mut step_bytes)?;
        let step = u64::from_le_bytes(step_bytes) as usize;

        let mut loss_bytes = [0u8; 4];
        reader.read_exact(&mut loss_bytes)?;
        let loss = f32::from_le_bytes(loss_bytes);

        if &magic == b"PRICHS02" {
            let mut num_params_bytes = [0u8; 8];
            reader.read_exact(&mut num_params_bytes)?;
            let num_params = u64::from_le_bytes(num_params_bytes) as usize;

            let mut params = self.get_parameter_tensors_mut();
            if num_params != params.len() {
                bail!("Checkpoint parameter count {} != model parameter count {}", num_params, params.len());
            }

            for (tensor, _) in params.iter_mut() {
                let mut len_bytes = [0u8; 8];
                reader.read_exact(&mut len_bytes)?;
                let len = u64::from_le_bytes(len_bytes) as usize;

                let mut data = vec![0.0f32; len];
                let raw_data = unsafe {
                    std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, len * 4)
                };
                reader.read_exact(raw_data)?;
                let shape = tensor.shape().clone();
                **tensor = Tensor::from_f32_slice(&data, shape)?;
            }
        } else if &magic == b"PRICHS_W" {
            let mut total_floats_bytes = [0u8; 8];
            reader.read_exact(&mut total_floats_bytes)?;

            let wte_len = self.config.vocab_size * self.config.hidden_dim;
            let mut wte_data = vec![0.0f32; wte_len];
            let raw_wte = unsafe {
                std::slice::from_raw_parts_mut(wte_data.as_mut_ptr() as *mut u8, wte_len * 4)
            };
            reader.read_exact(raw_wte)?;
            self.wte = Tensor::from_f32_slice(&wte_data, [self.config.vocab_size, self.config.hidden_dim])?;

            let norm_len = self.config.hidden_dim;
            let mut norm_data = vec![0.0f32; norm_len];
            let raw_norm = unsafe {
                std::slice::from_raw_parts_mut(norm_data.as_mut_ptr() as *mut u8, norm_len * 4)
            };
            if reader.read_exact(raw_norm).is_ok() {
                self.final_norm = Tensor::from_f32_slice(&norm_data, [self.config.hidden_dim])?;
            }
        } else {
            bail!("Invalid checkpoint magic in {:?}: {:?}", path, magic);
        }

        Ok((step, loss))
    }
}

/// Global Preallocated Model Gradients (Zero heap allocations during training)
pub struct ModelGradients {
    pub grads: Vec<Vec<f32>>,
}

impl ModelGradients {
    pub fn new(model: &PrichsModel) -> Self {
        let params = model.get_parameter_tensors();
        let mut grads = Vec::with_capacity(params.len());
        for (tensor, _) in params {
            let len = tensor.dims().iter().product::<usize>();
            grads.push(vec![0.0f32; len]);
        }
        Self { grads }
    }

    pub fn zero(&mut self) {
        self.grads.par_iter_mut().for_each(|g| {
            g.fill(0.0);
        });
    }

    pub fn scale(&mut self, factor: f32) {
        self.grads.par_iter_mut().for_each(|g| {
            for v in g.iter_mut() {
                *v *= factor;
            }
        });
    }

    pub fn add(&mut self, other: &ModelGradients) {
        self.grads
            .par_iter_mut()
            .zip(other.grads.par_iter())
            .for_each(|(g, og)| {
                for (dst, &src) in g.iter_mut().zip(og.iter()) {
                    *dst += src;
                }
            });
    }
}

pub struct ParamOptimizerState {
    pub exp_avg: Vec<f32>,
    pub exp_avg_sq: Vec<f32>,
    pub apply_weight_decay: bool,
}

/// In-place Multi-threaded AdamW Optimizer for all model parameters
pub struct FullModelOptimizer {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step_count: usize,
    pub states: Vec<ParamOptimizerState>,
}

impl FullModelOptimizer {
    pub fn new(model: &PrichsModel, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        let params = model.get_parameter_tensors();
        let mut states = Vec::with_capacity(params.len());
        for (tensor, apply_wd) in params {
            let len = tensor.dims().iter().product::<usize>();
            states.push(ParamOptimizerState {
                exp_avg: vec![0.0f32; len],
                exp_avg_sq: vec![0.0f32; len],
                apply_weight_decay: apply_wd,
            });
        }
        Self {
            beta1,
            beta2,
            eps,
            weight_decay,
            step_count: 0,
            states,
        }
    }

    pub fn step(&mut self, model: &mut PrichsModel, grads: &[Vec<f32>], lr: f32) -> Result<()> {
        self.step_count += 1;
        let t = self.step_count as i32;
        let bias_correction1 = 1.0 - self.beta1.powi(t);
        let bias_correction2 = 1.0 - self.beta2.powi(t);
        let step_size = lr / bias_correction1;

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let weight_decay = self.weight_decay;

        let mut params = model.get_parameter_tensors_mut();
        if params.len() != self.states.len() || params.len() != grads.len() {
            bail!("Mismatch in optimizer states count: {} vs {}", params.len(), self.states.len());
        }

        params
            .par_iter_mut()
            .zip(self.states.par_iter_mut())
            .zip(grads.par_iter())
            .for_each(|(((tensor, _), state), grad)| {
                if let Ok(tensor_data) = tensor.as_f32_mut_slice() {
                    let wd = if state.apply_weight_decay { weight_decay } else { 0.0 };
                    for i in 0..tensor_data.len().min(grad.len()) {
                        let g = grad[i];
                        let mut w = tensor_data[i];
                        if wd > 0.0 {
                            w -= lr * wd * w;
                        }
                        let m = beta1 * state.exp_avg[i] + (1.0 - beta1) * g;
                        let v = beta2 * state.exp_avg_sq[i] + (1.0 - beta2) * (g * g);
                        state.exp_avg[i] = m;
                        state.exp_avg_sq[i] = v;
                        let denom = (v / bias_correction2).sqrt() + eps;
                        tensor_data[i] = w - step_size * (m / denom);
                    }
                }
            });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> PrichsConfig {
        PrichsConfig {
            vocab_size: 128,
            hidden_dim: 64,
            intermediate_dim: 64,
            num_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: 2,
            max_seq_len: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            num_experts: 8,
            num_shared_experts: 1,
            num_experts_per_tok: 2,
            memory_concept_dim: 64,
            num_virtual_memory_tokens: 4,
        }
    }

    #[test]
    fn test_all_8_experts_and_router_receive_nonzero_gradients() {
        let config = create_test_config();
        let mut model = PrichsModel::new(config.clone()).expect("Failed to create model");
        let mut grads = ModelGradients::new(&model);

        // Sequence of 64 tokens ensures diverse router distribution across all 8 experts
        let inputs: Vec<u16> = (0..64).map(|i| (i % config.vocab_size as usize) as u16).collect();
        let targets: Vec<u16> = (1..65).map(|i| (i % config.vocab_size as usize) as u16).collect();

        let initial_bias = model.blocks[0].moe.router_bias.clone();

        let loss = model.forward_backward(&inputs, &targets, &mut grads)
            .expect("Forward-backward failed");

        assert!(loss > 0.0 && loss.is_finite(), "Loss must be finite and positive, got {}", loss);

        let num_experts = config.num_experts;
        let block_param_count = 10 + num_experts * 3;

        for l in 0..config.num_layers {
            let base_idx = 2 + l * block_param_count;

            // 1. Router W_router gradient must be NON-ZERO
            let router_grad = &grads.grads[base_idx + 6];
            let router_grad_norm: f32 = router_grad.iter().map(|g| g * g).sum();
            assert!(
                router_grad_norm > 1e-7,
                "Layer {} router gradient must be non-zero! Got norm {}",
                l,
                router_grad_norm
            );

            // 2. Shared Expert gradients must be NON-ZERO
            for p in 7..=9 {
                let shared_grad = &grads.grads[base_idx + p];
                let norm: f32 = shared_grad.iter().map(|g| g * g).sum();
                assert!(
                    norm > 1e-7,
                    "Layer {} Shared Expert param {} must have non-zero gradient! Got norm {}",
                    l,
                    p,
                    norm
                );
            }

            // 3. Check routed experts: each active expert must have non-zero gradients across gate, up, and down
            let mut active_experts_count = 0;
            for e in 0..num_experts {
                let gate_grad = &grads.grads[base_idx + 10 + e * 3 + 0];
                let up_grad = &grads.grads[base_idx + 10 + e * 3 + 1];
                let down_grad = &grads.grads[base_idx + 10 + e * 3 + 2];

                let g_norm: f32 = gate_grad.iter().map(|g| g * g).sum();
                let u_norm: f32 = up_grad.iter().map(|g| g * g).sum();
                let d_norm: f32 = down_grad.iter().map(|g| g * g).sum();

                if g_norm > 1e-8 || u_norm > 1e-8 || d_norm > 1e-8 {
                    assert!(g_norm > 1e-8 && u_norm > 1e-8 && d_norm > 1e-8,
                        "Expert {} at layer {} has inconsistent gradients: gate={}, up={}, down={}",
                        e, l, g_norm, u_norm, d_norm
                    );
                    active_experts_count += 1;
                }
            }

            // With 64 tokens and Top-2 routing across 8 experts, multiple experts must be active
            assert!(
                active_experts_count >= 6,
                "At least 6 of 8 experts must be engaged on diverse batch. Activated: {}/{}",
                active_experts_count,
                num_experts
            );

            // 4. Router bias must update dynamically via Loss-Free load balancing
            let updated_bias = &model.blocks[l].moe.router_bias;
            let bias_changed = initial_bias.iter().zip(updated_bias.iter()).any(|(b0, b1)| (b0 - b1).abs() > 1e-8);
            assert!(bias_changed, "Layer {} router_bias must update dynamically after step!", l);
        }
    }

    #[test]
    fn test_causal_attention_and_rope_backward_correctness() {
        let config = create_test_config();
        let mut model = PrichsModel::new(config.clone()).expect("Failed to create model");
        let mut grads = ModelGradients::new(&model);

        // Sequence of tokens with position-dependent causal dependency
        let inputs_a: Vec<u16> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let targets_a: Vec<u16> = vec![20, 30, 40, 50, 60, 70, 80, 90];

        let loss_a = model.forward_backward(&inputs_a, &targets_a, &mut grads)
            .expect("Forward-backward failed");
        assert!(loss_a > 0.0 && loss_a.is_finite());

        let base_idx = 2; // Layer 0
        // Check attention projection gradients are strictly non-zero
        let w_q_norm: f32 = grads.grads[base_idx + 1].iter().map(|g| g * g).sum();
        let w_k_norm: f32 = grads.grads[base_idx + 2].iter().map(|g| g * g).sum();
        let w_v_norm: f32 = grads.grads[base_idx + 3].iter().map(|g| g * g).sum();
        let w_o_norm: f32 = grads.grads[base_idx + 4].iter().map(|g| g * g).sum();

        assert!(w_q_norm > 1e-7, "W_q gradient must be non-zero! Got {}", w_q_norm);
        assert!(w_k_norm > 1e-7, "W_k gradient must be non-zero! Got {}", w_k_norm);
        assert!(w_v_norm > 1e-7, "W_v gradient must be non-zero! Got {}", w_v_norm);
        assert!(w_o_norm > 1e-7, "W_o gradient must be non-zero! Got {}", w_o_norm);

        // RoPE sensitivity check: permuting token order MUST yield different gradients
        let inputs_b: Vec<u16> = vec![80, 70, 60, 50, 40, 30, 20, 10];
        let targets_b: Vec<u16> = vec![70, 60, 50, 40, 30, 20, 10, 20];
        let mut grads_b = ModelGradients::new(&model);
        let _ = model.forward_backward(&inputs_b, &targets_b, &mut grads_b).unwrap();

        let w_q_diff: f32 = grads.grads[base_idx + 1].iter()
            .zip(grads_b.grads[base_idx + 1].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            w_q_diff > 1e-5,
            "RoPE must make attention order-dependent! Token permutation gradient difference was 0!"
        );

        // Exact analytical vs numerical finite-difference gradient check on SelfAttention
        let (cos_cache, sin_cache) = precompute_rope_cache(8, model.blocks[0].attention.head_dim, 10000.0);
        let x = Tensor::randn([8, config.hidden_dim], 0.0, 0.1).unwrap();
        let (out, stash) = model.blocks[0].attention.forward_with_rope(&x, &cos_cache, &sin_cache, 0).unwrap();

        let target = Tensor::randn([8, config.hidden_dim], 0.0, 0.1).unwrap();
        let target_data = target.as_f32_slice().unwrap();

        // Analytical gradient for W_o: dW_o = attn_concat^T * target
        let mut dw_o = vec![0.0f32; config.hidden_dim * config.hidden_dim];
        prichs_core::ops::matmul_at_b(&stash.attn_concat, target_data, &mut dw_o, 8, config.hidden_dim, config.hidden_dim).unwrap();

        let param_idx = 5;
        let analytical = dw_o[param_idx];

        let eps = 1e-3f32;
        let mut attn_plus = model.blocks[0].attention.clone();
        attn_plus.out_proj.as_f32_mut_slice().unwrap()[param_idx] += eps;
        let (out_plus, _) = attn_plus.forward_with_rope(&x, &cos_cache, &sin_cache, 0).unwrap();
        let mut loss_plus = 0.0f32;
        for i in 0..out.as_f32_slice().unwrap().len() {
            loss_plus += out_plus.as_f32_slice().unwrap()[i] * target_data[i];
        }

        let mut attn_minus = model.blocks[0].attention.clone();
        attn_minus.out_proj.as_f32_mut_slice().unwrap()[param_idx] -= eps;
        let (out_minus, _) = attn_minus.forward_with_rope(&x, &cos_cache, &sin_cache, 0).unwrap();
        let mut loss_minus = 0.0f32;
        for i in 0..out.as_f32_slice().unwrap().len() {
            loss_minus += out_minus.as_f32_slice().unwrap()[i] * target_data[i];
        }

        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        let rel_err = (analytical - numerical).abs() / (analytical.abs() + numerical.abs() + 1e-5);
        assert!(
            rel_err < 0.01,
            "Analytical gradient ({}) must match numerical gradient ({})! rel_err: {}",
            analytical,
            numerical,
            rel_err
        );
    }

    #[test]
    fn test_one_step_loss_decreases() {
        let config = create_test_config();
        let mut model = PrichsModel::new(config.clone()).expect("Failed to create model");
        let mut optimizer = FullModelOptimizer::new(&model, 0.9, 0.95, 1e-8, 0.0);
        let mut grads = ModelGradients::new(&model);

        let inputs: Vec<u16> = vec![5, 12, 19, 26, 33, 40, 47, 54];
        let targets: Vec<u16> = vec![12, 19, 26, 33, 40, 47, 54, 61];

        let lr = 0.005;

        // Step 1
        let loss1 = model.train_step(&inputs, &targets, &mut grads, &mut optimizer, lr)
            .expect("Step 1 failed");

        // Step 2
        let loss2 = model.train_step(&inputs, &targets, &mut grads, &mut optimizer, lr)
            .expect("Step 2 failed");

        // Step 3
        let loss3 = model.train_step(&inputs, &targets, &mut grads, &mut optimizer, lr)
            .expect("Step 3 failed");

        assert!(
            loss2 < loss1,
            "Loss must strictly decrease from step 1 ({:.4}) to step 2 ({:.4})",
            loss1,
            loss2
        );
        assert!(
            loss3 < loss2,
            "Loss must strictly decrease from step 2 ({:.4}) to step 3 ({:.4})",
            loss2,
            loss3
        );
    }

    #[test]
    fn test_persistent_vram_train_step_decreases_loss() {
        let mut cfg = PrichsConfig::mini();
        cfg.num_layers = 2;
        cfg.hidden_dim = 64;
        cfg.intermediate_dim = 64;
        cfg.vocab_size = 128;
        cfg.num_attention_heads = 2;
        cfg.num_kv_heads = 1;

        let mut model = PrichsModel::new(cfg).expect("Model init failed");
        let mut engine = match model.create_gpu_engine() {
            Ok(eng) => eng,
            Err(_) => return, // Skip test if no CUDA hardware in CI
        };

        let inputs: Vec<u16> = vec![5, 12, 19, 26, 33, 40, 47, 54];
        let targets: Vec<u16> = vec![12, 19, 26, 33, 40, 47, 54, 61];
        let mask = vec![1.0f32; 8];
        let lr = 0.005;

        // Step 1
        let loss1 = model.train_step_vram(&mut engine, &inputs, &targets, &mask, lr, 1)
            .expect("Step 1 failed");

        // Step 2
        let loss2 = model.train_step_vram(&mut engine, &inputs, &targets, &mask, lr, 2)
            .expect("Step 2 failed");

        // Step 3
        let loss3 = model.train_step_vram(&mut engine, &inputs, &targets, &mask, lr, 3)
            .expect("Step 3 failed");

        assert!(
            loss2 < loss1,
            "VRAM Loss must decrease: step 1 ({:.4}) -> step 2 ({:.4})",
            loss1,
            loss2
        );
        assert!(
            loss3 < loss2,
            "VRAM Loss must decrease: step 2 ({:.4}) -> step 3 ({:.4})",
            loss2,
            loss3
        );

        model.sync_weights_from_engine(&engine).expect("Sync weights failed");
        println!("Persistent VRAM Engine: Loss monotonically decreased: {:.4} -> {:.4} -> {:.4}", loss1, loss2, loss3);
    }
}
