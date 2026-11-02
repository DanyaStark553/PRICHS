use anyhow::Result;
use prichs_core::ops::{matmul_2d, swiglu, top_k_gating_unbiased};
use prichs_core::Tensor;

#[derive(Debug, Clone)]
pub struct Expert {
    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,
}

impl Expert {
    pub fn new(hidden_dim: usize, intermediate_dim: usize) -> Result<Self> {
        let std = (2.0 / (hidden_dim + intermediate_dim) as f32).sqrt();
        Ok(Self {
            gate_proj: Tensor::randn([hidden_dim, intermediate_dim], 0.0, std)?,
            up_proj: Tensor::randn([hidden_dim, intermediate_dim], 0.0, std)?,
            down_proj: Tensor::randn([intermediate_dim, hidden_dim], 0.0, std)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [seq_len, hidden_dim]
        let gate = matmul_2d(x, &self.gate_proj)?;
        let up = matmul_2d(x, &self.up_proj)?;
        let activated = swiglu(&gate, &up)?;
        matmul_2d(&activated, &self.down_proj)
    }
}

#[derive(Debug, Clone)]
pub struct ExpertForwardStash {
    pub expert_idx: usize,
    pub token_indices: Vec<usize>,
    pub weights: Vec<f32>,
    pub x: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub act: Vec<f32>,
    pub out: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct MoEStash {
    pub router_logits: Vec<f32>,
    pub selected_experts: Vec<usize>,
    pub gating_weights: Vec<f32>,
    pub shared_gate: Vec<f32>,
    pub shared_up: Vec<f32>,
    pub shared_act: Vec<f32>,
    pub shared_out: Vec<f32>,
    pub routed_stashes: Vec<ExpertForwardStash>,
}

/// Fine-Grained MoE Layer with Auxiliary-Loss-Free Dynamic Load Balancing
/// and Continuous StarkDB Concept Bias Injection.
#[derive(Debug, Clone)]
pub struct MoELayer {
    pub router: Tensor,              // [hidden_dim, num_experts]
    pub router_bias: Vec<f32>,       // Auxiliary-Loss-Free dynamic bias [num_experts]
    pub shared_expert: Expert,       // Always active (100% time - core identity anchor)
    pub experts: Vec<Expert>,        // Fine-Grained routed sparse experts
    pub top_k: usize,
}

impl MoELayer {
    pub fn new(
        hidden_dim: usize,
        intermediate_dim: usize,
        num_experts: usize,
        top_k: usize,
    ) -> Result<Self> {
        let std = (2.0 / (hidden_dim + num_experts) as f32).sqrt();
        let router = Tensor::randn([hidden_dim, num_experts], 0.0, std)?;
        let router_bias = vec![0.0f32; num_experts];
        let shared_expert = Expert::new(hidden_dim, intermediate_dim)?;

        let mut experts = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
            experts.push(Expert::new(hidden_dim, intermediate_dim)?);
        }

        Ok(Self {
            router,
            router_bias,
            shared_expert,
            experts,
            top_k,
        })
    }

    /// Full Batched Forward Pass with Activation Stashing for 100% Honest Backward Pass
    pub fn forward_train(
        &self,
        x: &Tensor,
        starkdb_concept_bias: Option<&[f32]>,
    ) -> Result<(Tensor, MoEStash)> {
        let seq_len = x.dims()[0];
        let hidden_dim = x.dims()[1];
        let num_experts = self.experts.len();

        // 1. Shared Expert Forward
        let shared_gate = matmul_2d(x, &self.shared_expert.gate_proj)?;
        let shared_up = matmul_2d(x, &self.shared_expert.up_proj)?;
        let shared_act = swiglu(&shared_gate, &shared_up)?;
        let shared_out = matmul_2d(&shared_act, &self.shared_expert.down_proj)?;

        let shared_gate_slice = shared_gate.as_f32_slice()?.to_vec();
        let shared_up_slice = shared_up.as_f32_slice()?.to_vec();
        let shared_act_slice = shared_act.as_f32_slice()?.to_vec();
        let shared_out_slice = shared_out.as_f32_slice()?.to_vec();

        // 2. Compute Router Logits: [seq_len, num_experts]
        let router_logits = matmul_2d(x, &self.router)?;
        let logits_data = router_logits.as_f32_slice()?;

        let mut selected_experts = vec![0usize; seq_len * self.top_k];
        let mut gating_weights = vec![0.0f32; seq_len * self.top_k];

        // Token binning for Batched execution: expert_idx -> list of (token_idx, weight)
        let mut expert_assignments: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];

        for t in 0..seq_len {
            let raw_logits = &logits_data[t * num_experts..(t + 1) * num_experts];
            let mut adjusted_logits = Vec::with_capacity(num_experts);
            for e in 0..num_experts {
                let mut logit = raw_logits[e] + self.router_bias[e];
                if let Some(concept_bias) = starkdb_concept_bias {
                    if e < concept_bias.len() {
                        logit += concept_bias[e];
                    }
                }
                adjusted_logits.push(logit);
            }

            let gating = top_k_gating_unbiased(raw_logits, &adjusted_logits, self.top_k);
            for (k, (&expert_idx, &weight)) in gating.indices.iter().zip(gating.weights.iter()).enumerate() {
                selected_experts[t * self.top_k + k] = expert_idx;
                gating_weights[t * self.top_k + k] = weight;
                expert_assignments[expert_idx].push((t, weight));
            }
        }

        // 3. Batched cuBLAS Execution for each Routed Expert (Gather-Scatter)
        let x_data = x.as_f32_slice()?;
        let mut total_out_data = shared_out_slice.clone();
        let mut routed_stashes = Vec::with_capacity(num_experts);

        for (e, assignments) in expert_assignments.into_iter().enumerate() {
            let n_tokens = assignments.len();
            if n_tokens == 0 {
                continue;
            }

            let mut batch_token_indices = Vec::with_capacity(n_tokens);
            let mut batch_weights = Vec::with_capacity(n_tokens);
            let mut x_batch = vec![0.0f32; n_tokens * hidden_dim];

            for (i, &(tok_idx, weight)) in assignments.iter().enumerate() {
                batch_token_indices.push(tok_idx);
                batch_weights.push(weight);
                let src_row = &x_data[tok_idx * hidden_dim..(tok_idx + 1) * hidden_dim];
                x_batch[i * hidden_dim..(i + 1) * hidden_dim].copy_from_slice(src_row);
            }

            let x_tensor = Tensor::from_f32_slice(&x_batch, [n_tokens, hidden_dim])?;
            let gate = matmul_2d(&x_tensor, &self.experts[e].gate_proj)?;
            let up = matmul_2d(&x_tensor, &self.experts[e].up_proj)?;
            let act = swiglu(&gate, &up)?;
            let exp_out = matmul_2d(&act, &self.experts[e].down_proj)?;

            let gate_slice = gate.as_f32_slice()?.to_vec();
            let up_slice = up.as_f32_slice()?.to_vec();
            let act_slice = act.as_f32_slice()?.to_vec();
            let out_slice = exp_out.as_f32_slice()?.to_vec();

            // Scatter-Add to total output
            for (i, (&tok_idx, &weight)) in batch_token_indices.iter().zip(batch_weights.iter()).enumerate() {
                let exp_row = &out_slice[i * hidden_dim..(i + 1) * hidden_dim];
                let out_row = &mut total_out_data[tok_idx * hidden_dim..(tok_idx + 1) * hidden_dim];
                for d in 0..hidden_dim {
                    out_row[d] += weight * exp_row[d];
                }
            }

            routed_stashes.push(ExpertForwardStash {
                expert_idx: e,
                token_indices: batch_token_indices,
                weights: batch_weights,
                x: x_batch,
                gate: gate_slice,
                up: up_slice,
                act: act_slice,
                out: out_slice,
            });
        }

        let out_tensor = Tensor::from_f32_slice(&total_out_data, [seq_len, hidden_dim])?;
        let stash = MoEStash {
            router_logits: logits_data.to_vec(),
            selected_experts,
            gating_weights,
            shared_gate: shared_gate_slice,
            shared_up: shared_up_slice,
            shared_act: shared_act_slice,
            shared_out: shared_out_slice,
            routed_stashes,
        };

        Ok((out_tensor, stash))
    }

    /// Fast Batched Forward pass with optional StarkDB memory concept bias
    pub fn forward_with_memory(
        &self,
        x: &Tensor,
        starkdb_concept_bias: Option<&[f32]>,
    ) -> Result<Tensor> {
        let (out, _) = self.forward_train(x, starkdb_concept_bias)?;
        Ok(out)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_memory(x, None)
    }

    /// Auxiliary-Loss-Free Dynamic Load Balancing Update
    pub fn update_aux_free_bias(
        &mut self,
        expert_token_counts: &[usize],
        total_tokens: usize,
        gamma: f32,
    ) {
        let num_experts = self.experts.len();
        if total_tokens == 0 || num_experts == 0 {
            return;
        }

        let target_load = (self.top_k as f32) / (num_experts as f32);
        for i in 0..num_experts {
            let actual_load = expert_token_counts[i] as f32 / total_tokens as f32;
            let diff = target_load - actual_load;
            self.router_bias[i] += gamma * diff;
        }
    }
}
