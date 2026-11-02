use crate::tensor::Tensor;
use anyhow::{bail, Result};
pub use prichs_cuda::{matmul_a_bt_gpu, matmul_at_b_gpu, matmul_gpu};
use rayon::prelude::*;

pub const GPU_DISPATCH_THRESHOLD: usize = 65_536;

pub fn matmul_2d(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_dims = a.dims();
    let b_dims = b.dims();
    if a_dims.len() != 2 || b_dims.len() != 2 {
        bail!("matmul_2d requires 2D tensors, got {:?} and {:?}", a_dims, b_dims);
    }
    let (m, k1) = (a_dims[0], a_dims[1]);
    let (k2, n) = (b_dims[0], b_dims[1]);
    if k1 != k2 {
        bail!("Incompatible shapes for matmul: [{}, {}] x [{}, {}]", m, k1, k2, n);
    }

    let a_data = a.as_f32_slice()?;
    let b_data = b.as_f32_slice()?;
    let mut out_data = vec![0.0f32; m * n];

    // 1. Smart Hybrid Dispatch:
    // Large batched GEMMs (m > 1 && m * k * n >= GPU_DISPATCH_THRESHOLD) can go to Hardware cuBLAS.
    // Single-token generation (m == 1) runs on CPU with zero PCIe latency and zero VRAM allocation!
    if m > 1 && m * k1 * n >= GPU_DISPATCH_THRESHOLD {
        if let Ok(()) = matmul_gpu(a_data, b_data, &mut out_data, m, k1, n) {
            return Tensor::from_f32_slice(&out_data, [m, n]);
        }
    }

    // 2. CPU Execution via Rayon
    if m == 1 {
        // Fast path for 1 x k1 (e.g. token-by-token MoE expert):
        let a_row = &a_data[0..k1];
        for k in 0..k1 {
            let a_val = a_row[k];
            let b_row = &b_data[k * n..(k + 1) * n];
            for j in 0..n {
                out_data[j] += a_val * b_row[j];
            }
        }
    } else {
        out_data
            .par_chunks_exact_mut(n)
            .enumerate()
            .for_each(|(i, out_row)| {
                let a_row = &a_data[i * k1..(i + 1) * k1];
                for k in 0..k1 {
                    let a_val = a_row[k];
                    let b_row = &b_data[k * n..(k + 1) * n];
                    for j in 0..n {
                        out_row[j] += a_val * b_row[j];
                    }
                }
            });
    }

    Tensor::from_f32_slice(&out_data, [m, n])
}

/// Smart Hybrid dW = A^T * B (Weight gradient)
pub fn matmul_at_b(a: &[f32], b: &[f32], dw: &mut [f32], m: usize, k: usize, n: usize) -> Result<()> {
    if m > 1 && k * m * n >= GPU_DISPATCH_THRESHOLD {
        if let Ok(()) = matmul_at_b_gpu(a, b, dw, m, k, n) {
            return Ok(());
        }
    }

    // CPU fallback with Rayon
    dw.par_chunks_exact_mut(n).enumerate().for_each(|(i, dw_row)| {
        for t in 0..m {
            let a_val = a[t * k + i];
            let b_row = &b[t * n..(t + 1) * n];
            for j in 0..n {
                dw_row[j] += a_val * b_row[j];
            }
        }
    });

    Ok(())
}

/// Smart Hybrid dX = dY * W^T (Input gradient and LM Head projection)
pub fn matmul_a_bt(dy: &[f32], w: &[f32], dx: &mut [f32], m: usize, k: usize, n: usize) -> Result<()> {
    if m > 1 && m * k * n >= GPU_DISPATCH_THRESHOLD {
        if let Ok(()) = matmul_a_bt_gpu(dy, w, dx, m, k, n) {
            return Ok(());
        }
    }

    // CPU fallback with Rayon
    dx.par_chunks_exact_mut(n).enumerate().for_each(|(i, dx_row)| {
        let dy_row = &dy[i * k..(i + 1) * k];
        for j in 0..n {
            let w_row = &w[j * k..(j + 1) * k];
            let mut dot = 0.0f32;
            for c in 0..k {
                dot += dy_row[c] * w_row[c];
            }
            dx_row[j] = dot;
        }
    });

    Ok(())
}

pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.dims();
    let hidden_dim = *dims.last().unwrap();
    let weight_data = weight.as_f32_slice()?;
    if weight_data.len() != hidden_dim {
        bail!("Weight length {} != hidden_dim {}", weight_data.len(), hidden_dim);
    }

    let x_data = x.as_f32_slice()?;
    let mut out_data = vec![0.0f32; x_data.len()];

    // Parallel RMSNorm across token rows
    out_data
        .par_chunks_exact_mut(hidden_dim)
        .zip(x_data.par_chunks_exact(hidden_dim))
        .for_each(|(out_row, row_slice)| {
            let sum_sq: f32 = row_slice.iter().map(|v| v * v).sum();
            let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;

            for c in 0..hidden_dim {
                out_row[c] = row_slice[c] * inv_rms * weight_data[c];
            }
        });

    Tensor::from_f32_slice(&out_data, x.shape().clone())
}

#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let gate_data = gate.as_f32_slice()?;
    let up_data = up.as_f32_slice()?;
    if gate_data.len() != up_data.len() {
        bail!("Gate and Up tensors must have identical length");
    }

    let mut out_data = vec![0.0f32; gate_data.len()];
    out_data
        .par_iter_mut()
        .zip(gate_data.par_iter())
        .zip(up_data.par_iter())
        .for_each(|((out, &g), &u)| {
            *out = silu(g) * u;
        });

    Tensor::from_f32_slice(&out_data, gate.shape().clone())
}

/// Stable Softmax along last dimension with Rayon parallelism
pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let last_dim = *dims.last().unwrap();
    let x_data = x.as_f32_slice()?;
    let mut out_data = vec![0.0f32; x_data.len()];

    out_data
        .par_chunks_exact_mut(last_dim)
        .zip(x_data.par_chunks_exact(last_dim))
        .for_each(|(out_row, row_slice)| {
            let max_val = row_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for c in 0..last_dim {
                let exp_val = (row_slice[c] - max_val).exp();
                out_row[c] = exp_val;
                sum_exp += exp_val;
            }
            let inv_sum = 1.0 / (sum_exp + 1e-9);
            for c in 0..last_dim {
                out_row[c] *= inv_sum;
            }
        });

    Tensor::from_f32_slice(&out_data, x.shape().clone())
}

/// Top-K Selection for MoE Router
#[derive(Debug, Clone)]
pub struct TopKResult {
    pub indices: Vec<usize>,
    pub weights: Vec<f32>,
}

pub fn top_k_gating(logits: &[f32], k: usize) -> TopKResult {
    assert!(k <= logits.len(), "k must be <= logits length");
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    // Sort descending by value
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_k = &indexed[..k];
    let max_val = top_k.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    let mut exp_weights = Vec::with_capacity(k);
    for &(_, v) in top_k {
        let e = (v - max_val).exp();
        exp_weights.push(e);
        sum_exp += e;
    }

    let inv_sum = 1.0 / (sum_exp + 1e-9);
    let weights: Vec<f32> = exp_weights.iter().map(|w| w * inv_sum).collect();
    let indices: Vec<usize> = top_k.iter().map(|(idx, _)| *idx).collect();

    TopKResult { indices, weights }
}

/// DeepSeek-V3 Invariant: Top-K routing with Bias, but Softmax weighting WITHOUT Bias
pub fn top_k_gating_unbiased(
    raw_logits: &[f32],
    routing_logits: &[f32],
    k: usize,
) -> TopKResult {
    assert!(k <= raw_logits.len(), "k must be <= logits length");
    assert_eq!(raw_logits.len(), routing_logits.len(), "raw and routing logits must match");

    // 1. Pick Top-K indices based purely on routing_logits (raw + bias)
    let mut indexed: Vec<(usize, f32)> = routing_logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_k_indices: Vec<usize> = indexed[..k].iter().map(|(idx, _)| *idx).collect();

    // 2. Compute Softmax weighting based STRICTLY on raw_logits (unbiased)
    let raw_top_k: Vec<f32> = top_k_indices.iter().map(|&idx| raw_logits[idx]).collect();
    let max_val = raw_top_k.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    let mut exp_weights = Vec::with_capacity(k);
    for &v in &raw_top_k {
        let e = (v - max_val).exp();
        exp_weights.push(e);
        sum_exp += e;
    }

    let inv_sum = 1.0 / (sum_exp + 1e-9);
    let weights: Vec<f32> = exp_weights.iter().map(|w| w * inv_sum).collect();

    TopKResult { indices: top_k_indices, weights }
}

pub fn add_tensors(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_data = a.as_f32_slice()?;
    let b_data = b.as_f32_slice()?;
    if a_data.len() != b_data.len() {
        bail!("Tensor length mismatch in addition: {} != {}", a_data.len(), b_data.len());
    }

    let mut out_data = vec![0.0f32; a_data.len()];
    out_data
        .par_iter_mut()
        .zip(a_data.par_iter())
        .zip(b_data.par_iter())
        .for_each(|((out, &av), &bv)| {
            *out = av + bv;
        });

    Tensor::from_f32_slice(&out_data, a.shape().clone())
}
