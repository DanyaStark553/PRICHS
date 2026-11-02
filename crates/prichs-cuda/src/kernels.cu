#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__inline__ __device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceMax(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceMax(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : -1e30f;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

extern "C" {

// =========================================================================
// 1. ADAMW FUSED UPDATE KERNEL (VECTORIZED float4 for 300+ GB/s VRAM THROUGHPUT)
// =========================================================================

__global__ void adamw_kernel_vec4(
    float4* __restrict__ weights,
    float4* __restrict__ grads,
    float4* __restrict__ exp_avg,
    float4* __restrict__ exp_avg_sq,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float bias_correction1,
    const float bias_correction2,
    const size_t total_vec4
) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_vec4) return;

    float4 g = grads[idx];
    float4 w = weights[idx];
    float4 m = exp_avg[idx];
    float4 v = exp_avg_sq[idx];

    // Element 0 (x)
    if (weight_decay > 0.0f) w.x -= lr * weight_decay * w.x;
    m.x = beta1 * m.x + (1.0f - beta1) * g.x;
    v.x = beta2 * v.x + (1.0f - beta2) * (g.x * g.x);
    w.x -= lr * (m.x / bias_correction1) / (sqrtf(v.x / bias_correction2) + eps);

    // Element 1 (y)
    if (weight_decay > 0.0f) w.y -= lr * weight_decay * w.y;
    m.y = beta1 * m.y + (1.0f - beta1) * g.y;
    v.y = beta2 * v.y + (1.0f - beta2) * (g.y * g.y);
    w.y -= lr * (m.y / bias_correction1) / (sqrtf(v.y / bias_correction2) + eps);

    // Element 2 (z)
    if (weight_decay > 0.0f) w.z -= lr * weight_decay * w.z;
    m.z = beta1 * m.z + (1.0f - beta1) * g.z;
    v.z = beta2 * v.z + (1.0f - beta2) * (g.z * g.z);
    w.z -= lr * (m.z / bias_correction1) / (sqrtf(v.z / bias_correction2) + eps);

    // Element 3 (w)
    if (weight_decay > 0.0f) w.w -= lr * weight_decay * w.w;
    m.w = beta1 * m.w + (1.0f - beta1) * g.w;
    v.w = beta2 * v.w + (1.0f - beta2) * (g.w * g.w);
    w.w -= lr * (m.w / bias_correction1) / (sqrtf(v.w / bias_correction2) + eps);

    // 128-bit Burst Write Back
    weights[idx] = w;
    exp_avg[idx] = m;
    exp_avg_sq[idx] = v;
    grads[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__global__ void adamw_kernel_tail(
    float* __restrict__ weights,
    float* __restrict__ grads,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float bias_correction1,
    const float bias_correction2,
    const size_t offset,
    const size_t total_elements
) {
    size_t idx = offset + threadIdx.x;
    if (idx >= total_elements) return;

    float g = grads[idx];
    float w = weights[idx];

    if (weight_decay > 0.0f) {
        w -= lr * weight_decay * w;
    }

    float m = beta1 * exp_avg[idx] + (1.0f - beta1) * g;
    float v = beta2 * exp_avg_sq[idx] + (1.0f - beta2) * (g * g);

    exp_avg[idx] = m;
    exp_avg_sq[idx] = v;

    float m_hat = m / bias_correction1;
    float v_hat = v / bias_correction2;
    float step = m_hat / (sqrtf(v_hat) + eps);

    weights[idx] = w - lr * step;
    grads[idx] = 0.0f;
}

void launch_adamw_gpu(
    float* dev_weights,
    float* dev_grads,
    float* dev_exp_avg,
    float* dev_exp_avg_sq,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int step,
    size_t total_elements,
    cudaStream_t stream
) {
    if (total_elements == 0) return;

    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);

    size_t total_vec4 = total_elements / 4;
    const int block_size = 256;

    if (total_vec4 > 0) {
        int grid_vec4 = (int)((total_vec4 + block_size - 1) / block_size);
        adamw_kernel_vec4<<<grid_vec4, block_size, 0, stream>>>(
            (float4*)dev_weights,
            (float4*)dev_grads,
            (float4*)dev_exp_avg,
            (float4*)dev_exp_avg_sq,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
            total_vec4
        );
    }

    size_t remainder = total_elements % 4;
    if (remainder > 0) {
        size_t offset = total_vec4 * 4;
        adamw_kernel_tail<<<1, 4, 0, stream>>>(
            dev_weights,
            dev_grads,
            dev_exp_avg,
            dev_exp_avg_sq,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
            offset,
            total_elements
        );
    }
}

// =========================================================================
// 2. INPUT TOKEN EMBEDDING GRADIENT ACCUMULATION (ATOMIC)
// =========================================================================

__global__ void accumulate_wte_grad_kernel(
    float* __restrict__ dev_grads,
    const unsigned short* __restrict__ dev_tokens,
    const float* __restrict__ dev_dx,
    size_t seq_len,
    size_t hidden_dim
) {
    size_t tid = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= seq_len * hidden_dim) return;
    size_t t = tid / hidden_dim;
    size_t d = tid % hidden_dim;
    unsigned short tok = dev_tokens[t];
    atomicAdd(&dev_grads[(size_t)tok * hidden_dim + d], dev_dx[tid]);
}

void launch_accumulate_wte_grad(
    float* dev_grads,
    const unsigned short* dev_tokens,
    const float* dev_dx,
    size_t seq_len,
    size_t hidden_dim,
    void* stream
) {
    if (seq_len == 0 || hidden_dim == 0) return;
    size_t total = seq_len * hidden_dim;
    const int block_size = 256;
    int grid = (int)((total + block_size - 1) / block_size);
    cudaStream_t s = (cudaStream_t)stream;
    accumulate_wte_grad_kernel<<<grid, block_size, 0, s>>>(
        dev_grads, dev_tokens, dev_dx, seq_len, hidden_dim
    );
}

// =========================================================================
// 3. FUSED SOFTMAX CROSS-ENTROPY FORWARD & BACKWARD (100% IN VRAM)
// =========================================================================

__global__ void fused_cross_entropy_kernel(
    const float* __restrict__ logits,
    const unsigned short* __restrict__ targets,
    const float* __restrict__ mask,
    float* __restrict__ d_logits,
    float* __restrict__ loss_out,
    size_t seq_len,
    size_t vocab_size,
    float active_count,
    float grad_scale
) {
    size_t t = blockIdx.x;
    if (t >= seq_len) return;

    float m_val = mask[t];
    unsigned short tgt = targets[t];
    const float* logits_row = logits + t * vocab_size;
    float* d_logits_row = d_logits + t * vocab_size;

    // 1. Find max
    float thread_max = -1e30f;
    for (size_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        thread_max = fmaxf(thread_max, logits_row[v]);
    }
    float row_max = blockReduceMax(thread_max);
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = row_max;
    __syncthreads();
    row_max = s_max;

    // 2. Sum exp
    float thread_sum = 0.0f;
    for (size_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        thread_sum += __expf(logits_row[v] - row_max);
    }
    float row_sum = blockReduceSum(thread_sum);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = row_sum;
    __syncthreads();
    row_sum = s_sum;

    float inv_sum = 1.0f / (row_sum + 1e-9f);

    // 3. Loss & Gradient
    if (threadIdx.x == 0 && m_val > 0.0f && active_count > 0.0f) {
        float tgt_logit = logits_row[tgt];
        float token_loss = - (tgt_logit - row_max) + logf(row_sum);
        atomicAdd(loss_out, token_loss / active_count);
    }

    float scale = (m_val > 0.0f && active_count > 0.0f) ? (grad_scale / active_count) : 0.0f;
    for (size_t v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        float p = __expf(logits_row[v] - row_max) * inv_sum;
        float target_delta = (v == tgt) ? 1.0f : 0.0f;
        d_logits_row[v] = scale * (p - target_delta);
    }
}

void launch_fused_cross_entropy(
    const float* dev_logits,
    const unsigned short* dev_targets,
    const float* dev_mask,
    float* dev_d_logits,
    float* dev_loss_out,
    size_t seq_len,
    size_t vocab_size,
    float active_count,
    float grad_scale,
    void* stream
) {
    if (seq_len == 0 || vocab_size == 0) return;
    const int block_size = 256;
    int grid = (int)seq_len;
    cudaStream_t s = (cudaStream_t)stream;
    fused_cross_entropy_kernel<<<grid, block_size, 0, s>>>(
        dev_logits, dev_targets, dev_mask, dev_d_logits, dev_loss_out,
        seq_len, vocab_size, active_count, grad_scale
    );
}

// =========================================================================
// 4. RMSNORM FORWARD & BACKWARD IN VRAM
// =========================================================================

__global__ void rmsnorm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    float* __restrict__ rstd_out,
    size_t seq_len,
    size_t hidden_dim,
    float eps
) {
    size_t t = blockIdx.x;
    if (t >= seq_len) return;

    const float* x_row = x + t * hidden_dim;
    float* out_row = out + t * hidden_dim;

    float thread_sum_sq = 0.0f;
    for (size_t d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
        float val = x_row[d];
        thread_sum_sq += val * val;
    }
    float row_sum_sq = blockReduceSum(thread_sum_sq);
    __shared__ float s_rstd;
    if (threadIdx.x == 0) {
        float mean_sq = row_sum_sq / (float)hidden_dim;
        s_rstd = rsqrtf(mean_sq + eps);
        rstd_out[t] = s_rstd;
    }
    __syncthreads();
    float rstd = s_rstd;

    for (size_t d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
        out_row[d] = x_row[d] * rstd * weight[d];
    }
}

__global__ void rmsnorm_backward_kernel(
    const float* __restrict__ dout,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ rstd_in,
    float* __restrict__ dx,
    float* __restrict__ dweight_accum,
    size_t seq_len,
    size_t hidden_dim
) {
    size_t t = blockIdx.x;
    if (t >= seq_len) return;

    const float* dout_row = dout + t * hidden_dim;
    const float* x_row = x + t * hidden_dim;
    float* dx_row = dx + t * hidden_dim;
    float rstd = rstd_in[t];

    float thread_sum_dout_x_w = 0.0f;
    for (size_t d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
        float w = weight[d];
        float val = dout_row[d] * x_row[d] * w;
        thread_sum_dout_x_w += val;
        atomicAdd(&dweight_accum[d], dout_row[d] * x_row[d] * rstd);
    }

    float row_sum = blockReduceSum(thread_sum_dout_x_w);
    __shared__ float s_factor;
    if (threadIdx.x == 0) {
        s_factor = row_sum * (rstd * rstd * rstd / (float)hidden_dim);
    }
    __syncthreads();
    float factor = s_factor;

    for (size_t d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
        dx_row[d] = dout_row[d] * weight[d] * rstd - x_row[d] * factor;
    }
}

void launch_rmsnorm_forward(
    const float* dev_x,
    const float* dev_weight,
    float* dev_out,
    float* dev_rstd_out,
    size_t seq_len,
    size_t hidden_dim,
    float eps,
    void* stream
) {
    const int block_size = 256;
    int grid = (int)seq_len;
    cudaStream_t s = (cudaStream_t)stream;
    rmsnorm_forward_kernel<<<grid, block_size, 0, s>>>(
        dev_x, dev_weight, dev_out, dev_rstd_out, seq_len, hidden_dim, eps
    );
}

void launch_rmsnorm_backward(
    const float* dev_dout,
    const float* dev_x,
    const float* dev_weight,
    const float* dev_rstd_in,
    float* dev_dx,
    float* dev_dweight_accum,
    size_t seq_len,
    size_t hidden_dim,
    void* stream
) {
    const int block_size = 256;
    int grid = (int)seq_len;
    cudaStream_t s = (cudaStream_t)stream;
    rmsnorm_backward_kernel<<<grid, block_size, 0, s>>>(
        dev_dout, dev_x, dev_weight, dev_rstd_in, dev_dx, dev_dweight_accum, seq_len, hidden_dim
    );
}

// =========================================================================
// 5. SWIGLU FORWARD & BACKWARD IN VRAM
// =========================================================================

__global__ void swiglu_forward_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    size_t total_elements
) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_elements) return;

    float g = gate[idx];
    float u = up[idx];
    float sig_g = 1.0f / (1.0f + __expf(-g));
    out[idx] = (g * sig_g) * u;
}

__global__ void swiglu_backward_kernel(
    const float* __restrict__ dact,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ dgate,
    float* __restrict__ dup,
    size_t total_elements
) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_elements) return;

    float g = gate[idx];
    float u = up[idx];
    float dy = dact[idx];

    float sig_g = 1.0f / (1.0f + __expf(-g));
    float silu_g = g * sig_g;
    float d_silu_g = sig_g * (1.0f + g * (1.0f - sig_g));

    dup[idx] = dy * silu_g;
    dgate[idx] = dy * u * d_silu_g;
}

void launch_swiglu_forward(
    const float* dev_gate,
    const float* dev_up,
    float* dev_out,
    size_t total_elements,
    void* stream
) {
    if (total_elements == 0) return;
    const int block_size = 256;
    int grid = (int)((total_elements + block_size - 1) / block_size);
    cudaStream_t s = (cudaStream_t)stream;
    swiglu_forward_kernel<<<grid, block_size, 0, s>>>(
        dev_gate, dev_up, dev_out, total_elements
    );
}

void launch_swiglu_backward(
    const float* dev_dact,
    const float* dev_gate,
    const float* dev_up,
    float* dev_dgate,
    float* dev_dup,
    size_t total_elements,
    void* stream
) {
    if (total_elements == 0) return;
    const int block_size = 256;
    int grid = (int)((total_elements + block_size - 1) / block_size);
    cudaStream_t s = (cudaStream_t)stream;
    swiglu_backward_kernel<<<grid, block_size, 0, s>>>(
        dev_dact, dev_gate, dev_up, dev_dgate, dev_dup, total_elements
    );
}

// =========================================================================
// 6. ROPE ROTARY POSITION EMBEDDING IN VRAM
// =========================================================================

__global__ void rope_kernel(
    float* __restrict__ q_or_k,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    int backward_sign
) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    size_t half_dim = head_dim / 2;
    size_t total_pairs = seq_len * num_heads * half_dim;
    if (idx >= total_pairs) return;

    size_t pair_idx = idx % half_dim;
    size_t temp = idx / half_dim;
    size_t head_idx = temp % num_heads;
    size_t token_idx = temp / num_heads;

    size_t base_offset = token_idx * (num_heads * head_dim) + head_idx * head_dim;
    size_t idx0 = base_offset + pair_idx * 2;
    size_t idx1 = idx0 + 1;

    size_t cache_idx = token_idx * half_dim + pair_idx;
    float cos_val = cos_cache[cache_idx];
    float sin_val = (float)backward_sign * sin_cache[cache_idx];

    float v0 = q_or_k[idx0];
    float v1 = q_or_k[idx1];

    q_or_k[idx0] = v0 * cos_val - v1 * sin_val;
    q_or_k[idx1] = v0 * sin_val + v1 * cos_val;
}

void launch_rope(
    float* dev_q_or_k,
    const float* dev_cos_cache,
    const float* dev_sin_cache,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    int backward_sign,
    void* stream
) {
    size_t total_pairs = seq_len * num_heads * (head_dim / 2);
    if (total_pairs == 0) return;
    const int block_size = 256;
    int grid = (int)((total_pairs + block_size - 1) / block_size);
    cudaStream_t s = (cudaStream_t)stream;
    rope_kernel<<<grid, block_size, 0, s>>>(
        dev_q_or_k, dev_cos_cache, dev_sin_cache, seq_len, num_heads, head_dim, backward_sign
    );
}

// =========================================================================
// 7. CAUSAL SOFTMAX FORWARD & BACKWARD IN VRAM
// =========================================================================

__global__ void causal_softmax_forward_kernel(
    float* __restrict__ scores,
    float scale,
    size_t num_heads,
    size_t seq_len
) {
    size_t h = blockIdx.y;
    size_t i = blockIdx.x;
    if (h >= num_heads || i >= seq_len) return;

    float* row = scores + (h * seq_len + i) * seq_len;

    float thread_max = -1e30f;
    for (size_t j = threadIdx.x; j <= i; j += blockDim.x) {
        thread_max = fmaxf(thread_max, row[j] * scale);
    }
    float row_max = blockReduceMax(thread_max);
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = row_max;
    __syncthreads();
    row_max = s_max;

    float thread_sum = 0.0f;
    for (size_t j = threadIdx.x; j <= i; j += blockDim.x) {
        thread_sum += __expf(row[j] * scale - row_max);
    }
    float row_sum = blockReduceSum(thread_sum);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = row_sum;
    __syncthreads();
    row_sum = s_sum;

    float inv_sum = 1.0f / (row_sum + 1e-9f);

    for (size_t j = threadIdx.x; j < seq_len; j += blockDim.x) {
        if (j <= i) {
            row[j] = __expf(row[j] * scale - row_max) * inv_sum;
        } else {
            row[j] = 0.0f;
        }
    }
}

__global__ void causal_softmax_backward_kernel(
    const float* __restrict__ d_attn,
    const float* __restrict__ probs,
    float* __restrict__ d_scores,
    float scale,
    size_t num_heads,
    size_t seq_len
) {
    size_t h = blockIdx.y;
    size_t i = blockIdx.x;
    if (h >= num_heads || i >= seq_len) return;

    size_t offset = (h * seq_len + i) * seq_len;
    const float* d_row = d_attn + offset;
    const float* p_row = probs + offset;
    float* ds_row = d_scores + offset;

    float thread_sum_dp = 0.0f;
    for (size_t j = threadIdx.x; j <= i; j += blockDim.x) {
        thread_sum_dp += d_row[j] * p_row[j];
    }
    float row_sum_dp = blockReduceSum(thread_sum_dp);
    __shared__ float s_sum_dp;
    if (threadIdx.x == 0) s_sum_dp = row_sum_dp;
    __syncthreads();
    float sum_dp = s_sum_dp;

    for (size_t j = threadIdx.x; j < seq_len; j += blockDim.x) {
        if (j <= i) {
            ds_row[j] = p_row[j] * (d_row[j] - sum_dp) * scale;
        } else {
            ds_row[j] = 0.0f;
        }
    }
}

void launch_causal_softmax_forward(
    float* dev_scores,
    float scale,
    size_t num_heads,
    size_t seq_len,
    void* stream
) {
    dim3 grid((unsigned int)seq_len, (unsigned int)num_heads);
    const int block_size = 256;
    cudaStream_t s = (cudaStream_t)stream;
    causal_softmax_forward_kernel<<<grid, block_size, 0, s>>>(
        dev_scores, scale, num_heads, seq_len
    );
}

void launch_causal_softmax_backward(
    const float* dev_d_attn,
    const float* dev_probs,
    float* dev_d_scores,
    float scale,
    size_t num_heads,
    size_t seq_len,
    void* stream
) {
    dim3 grid((unsigned int)seq_len, (unsigned int)num_heads);
    const int block_size = 256;
    cudaStream_t s = (cudaStream_t)stream;
    causal_softmax_backward_kernel<<<grid, block_size, 0, s>>>(
        dev_d_attn, dev_probs, dev_d_scores, scale, num_heads, seq_len
    );
}

// =========================================================================
// 8. TENSOR PERMUTATION & RESHAPE KERNELS FOR BATCHED ATTENTION
// =========================================================================

__global__ void transpose_0213_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim
) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    size_t total = seq_len * num_heads * head_dim;
    if (idx >= total) return;

    size_t d = idx % head_dim;
    size_t temp = idx / head_dim;
    size_t h = temp % num_heads;
    size_t t = temp / num_heads;

    size_t out_idx = (h * seq_len + t) * head_dim + d;
    out[out_idx] = in[idx];
}

__global__ void transpose_102_to_012_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    size_t num_heads,
    size_t seq_len,
    size_t head_dim
) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    size_t total = num_heads * seq_len * head_dim;
    if (idx >= total) return;

    size_t d = idx % head_dim;
    size_t temp = idx / head_dim;
    size_t t = temp % seq_len;
    size_t h = temp / seq_len;

    size_t out_idx = (t * num_heads + h) * head_dim + d;
    out[out_idx] = in[idx];
}

__global__ void repeat_kv_heads_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    size_t seq_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim
) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    size_t total = seq_len * num_heads * head_dim;
    if (idx >= total) return;

    size_t d = idx % head_dim;
    size_t temp = idx / head_dim;
    size_t h = temp % num_heads;
    size_t t = temp / num_heads;

    size_t kv_group = num_heads / num_kv_heads;
    size_t kv_h = h / kv_group;

    size_t in_idx = (t * num_kv_heads + kv_h) * head_dim + d;
    size_t out_idx = (h * seq_len + t) * head_dim + d;
    out[out_idx] = in[in_idx];
}

__global__ void reduce_kv_heads_grad_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    size_t seq_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim
) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    size_t total = seq_len * num_kv_heads * head_dim;
    if (idx >= total) return;

    size_t d = idx % head_dim;
    size_t temp = idx / head_dim;
    size_t kv_h = temp % num_kv_heads;
    size_t t = temp / num_kv_heads;

    size_t kv_group = num_heads / num_kv_heads;
    float sum = 0.0f;
    for (size_t g = 0; g < kv_group; g++) {
        size_t h = kv_h * kv_group + g;
        size_t in_idx = (h * seq_len + t) * head_dim + d;
        sum += in[in_idx];
    }
    size_t out_idx = (t * num_kv_heads + kv_h) * head_dim + d;
    out[out_idx] = sum;
}

__global__ void vector_add_kernel(
    const float* __restrict__ a,
    float* __restrict__ b,
    size_t total
) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total) {
        b[idx] += a[idx];
    }
}

void launch_transpose_0213(
    const float* dev_in,
    float* dev_out,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    void* stream
) {
    size_t total = seq_len * num_heads * head_dim;
    if (total == 0) return;
    const int block_size = 256;
    int grid = (int)((total + block_size - 1) / block_size);
    cudaStream_t s = (cudaStream_t)stream;
    transpose_0213_kernel<<<grid, block_size, 0, s>>>(
        dev_in, dev_out, seq_len, num_heads, head_dim
    );
}

void launch_transpose_102_to_012(
    const float* dev_in,
    float* dev_out,
    size_t num_heads,
    size_t seq_len,
    size_t head_dim,
    void* stream
) {
    size_t total = num_heads * seq_len * head_dim;
    if (total == 0) return;
    const int block_size = 256;
    int grid = (int)((total + block_size - 1) / block_size);
    cudaStream_t s = (cudaStream_t)stream;
    transpose_102_to_012_kernel<<<grid, block_size, 0, s>>>(
        dev_in, dev_out, num_heads, seq_len, head_dim
    );
}

void launch_repeat_kv_heads(
    const float* dev_in,
    float* dev_out,
    size_t seq_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    void* stream
) {
    size_t total = seq_len * num_heads * head_dim;
    if (total == 0) return;
    const int block_size = 256;
    int grid = (int)((total + block_size - 1) / block_size);
    cudaStream_t s = (cudaStream_t)stream;
    repeat_kv_heads_kernel<<<grid, block_size, 0, s>>>(
        dev_in, dev_out, seq_len, num_heads, num_kv_heads, head_dim
    );
}

void launch_reduce_kv_heads_grad(
    const float* dev_in,
    float* dev_out,
    size_t seq_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    void* stream
) {
    size_t total = seq_len * num_kv_heads * head_dim;
    if (total == 0) return;
    const int block_size = 256;
    int grid = (int)((total + block_size - 1) / block_size);
    cudaStream_t s = (cudaStream_t)stream;
    reduce_kv_heads_grad_kernel<<<grid, block_size, 0, s>>>(
        dev_in, dev_out, seq_len, num_heads, num_kv_heads, head_dim
    );
}

void launch_vector_add(
    const float* dev_a,
    float* dev_b,
    size_t total,
    void* stream
) {
    if (total == 0) return;
    const int block_size = 256;
    int grid = (int)((total + block_size - 1) / block_size);
    cudaStream_t s = (cudaStream_t)stream;
    vector_add_kernel<<<grid, block_size, 0, s>>>(dev_a, dev_b, total);
}

// =========================================================================
// 9. MOE GATHER, SCATTER & ROUTER GPU KERNELS (ZERO-CPU ACTIVATIONS)
// =========================================================================

__global__ void moe_gather_kernel(
    const float* __restrict__ in,
    const int* __restrict__ token_indices,
    float* __restrict__ out,
    size_t n_tokens,
    size_t hidden_dim
) {
    size_t tid = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    size_t total = n_tokens * hidden_dim;
    if (tid >= total) return;

    size_t i = tid / hidden_dim;
    size_t d = tid % hidden_dim;
    int src_tok = token_indices[i];

    out[tid] = in[(size_t)src_tok * hidden_dim + d];
}

__global__ void moe_scatter_add_kernel(
    const float* __restrict__ exp_out,
    const int* __restrict__ token_indices,
    const float* __restrict__ weights,
    float* __restrict__ accum_out,
    size_t n_tokens,
    size_t hidden_dim
) {
    size_t tid = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    size_t total = n_tokens * hidden_dim;
    if (tid >= total) return;

    size_t i = tid / hidden_dim;
    size_t d = tid % hidden_dim;
    int tok_idx = token_indices[i];
    float w = weights[i];

    atomicAdd(&accum_out[(size_t)tok_idx * hidden_dim + d], w * exp_out[tid]);
}

__global__ void moe_backward_gather_weighted_kernel(
    const float* __restrict__ d_moe,
    const int* __restrict__ token_indices,
    const float* __restrict__ weights,
    float* __restrict__ d_y_e,
    size_t n_tokens,
    size_t hidden_dim
) {
    size_t tid = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    size_t total = n_tokens * hidden_dim;
    if (tid >= total) return;

    size_t i = tid / hidden_dim;
    size_t d = tid % hidden_dim;
    int tok_idx = token_indices[i];
    float w = weights[i];

    d_y_e[tid] = w * d_moe[(size_t)tok_idx * hidden_dim + d];
}

__global__ void moe_router_dot_kernel(
    const float* __restrict__ exp_out,
    const float* __restrict__ d_moe,
    const int* __restrict__ token_indices,
    float* __restrict__ scalars_out,
    size_t n_tokens,
    size_t hidden_dim
) {
    size_t i = blockIdx.x;
    if (i >= n_tokens) return;

    int tok_idx = token_indices[i];
    const float* exp_row = exp_out + i * hidden_dim;
    const float* d_moe_row = d_moe + (size_t)tok_idx * hidden_dim;

    float thread_sum = 0.0f;
    for (size_t d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
        thread_sum += exp_row[d] * d_moe_row[d];
    }
    float row_sum = blockReduceSum(thread_sum);
    if (threadIdx.x == 0) {
        scalars_out[i] = row_sum;
    }
}

__global__ void moe_scatter_add_grad_kernel(
    const float* __restrict__ d_x_e,
    const float* __restrict__ d_x_e_tmp,
    const int* __restrict__ token_indices,
    const float* __restrict__ weights,
    float* __restrict__ d_normed2,
    size_t n_tokens,
    size_t hidden_dim
) {
    size_t tid = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    size_t total = n_tokens * hidden_dim;
    if (tid >= total) return;

    size_t i = tid / hidden_dim;
    size_t d = tid % hidden_dim;
    int tok_idx = token_indices[i];
    float w = weights[i];

    float val = w * (d_x_e[tid] + d_x_e_tmp[tid]);
    atomicAdd(&d_normed2[(size_t)tok_idx * hidden_dim + d], val);
}

void launch_moe_gather(
    const float* dev_in,
    const int* dev_token_indices,
    float* dev_out,
    size_t n_tokens,
    size_t hidden_dim,
    void* stream
) {
    size_t total = n_tokens * hidden_dim;
    if (total == 0) return;
    const int block_size = 256;
    int grid = (int)((total + block_size - 1) / block_size);
    cudaStream_t s = (cudaStream_t)stream;
    moe_gather_kernel<<<grid, block_size, 0, s>>>(dev_in, dev_token_indices, dev_out, n_tokens, hidden_dim);
}

void launch_moe_scatter_add(
    const float* dev_exp_out,
    const int* dev_token_indices,
    const float* dev_weights,
    float* dev_accum_out,
    size_t n_tokens,
    size_t hidden_dim,
    void* stream
) {
    size_t total = n_tokens * hidden_dim;
    if (total == 0) return;
    const int block_size = 256;
    int grid = (int)((total + block_size - 1) / block_size);
    cudaStream_t s = (cudaStream_t)stream;
    moe_scatter_add_kernel<<<grid, block_size, 0, s>>>(dev_exp_out, dev_token_indices, dev_weights, dev_accum_out, n_tokens, hidden_dim);
}

void launch_moe_backward_gather_weighted(
    const float* dev_d_moe,
    const int* dev_token_indices,
    const float* dev_weights,
    float* dev_d_y_e,
    size_t n_tokens,
    size_t hidden_dim,
    void* stream
) {
    size_t total = n_tokens * hidden_dim;
    if (total == 0) return;
    const int block_size = 256;
    int grid = (int)((total + block_size - 1) / block_size);
    cudaStream_t s = (cudaStream_t)stream;
    moe_backward_gather_weighted_kernel<<<grid, block_size, 0, s>>>(dev_d_moe, dev_token_indices, dev_weights, dev_d_y_e, n_tokens, hidden_dim);
}

void launch_moe_router_dot(
    const float* dev_exp_out,
    const float* dev_d_moe,
    const int* dev_token_indices,
    float* dev_scalars_out,
    size_t n_tokens,
    size_t hidden_dim,
    void* stream
) {
    if (n_tokens == 0) return;
    const int block_size = 256;
    int grid = (int)n_tokens;
    cudaStream_t s = (cudaStream_t)stream;
    moe_router_dot_kernel<<<grid, block_size, 0, s>>>(dev_exp_out, dev_d_moe, dev_token_indices, dev_scalars_out, n_tokens, hidden_dim);
}

void launch_moe_scatter_add_grad(
    const float* dev_d_x_e,
    const float* dev_d_x_e_tmp,
    const int* dev_token_indices,
    const float* dev_weights,
    float* dev_d_normed2,
    size_t n_tokens,
    size_t hidden_dim,
    void* stream
) {
    size_t total = n_tokens * hidden_dim;
    if (total == 0) return;
    const int block_size = 256;
    int grid = (int)((total + block_size - 1) / block_size);
    cudaStream_t s = (cudaStream_t)stream;
    moe_scatter_add_grad_kernel<<<grid, block_size, 0, s>>>(dev_d_x_e, dev_d_x_e_tmp, dev_token_indices, dev_weights, dev_d_normed2, n_tokens, hidden_dim);
}

} // extern "C"
