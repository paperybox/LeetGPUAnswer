import torch
import triton
import triton.language as tl


@triton.jit
def softmax_attention(Q_ptr, K_ptr, V_ptr, OUT_ptr,
                      M, N, d,
                      Q_stride_M, Q_stride_d,
                      K_stride_N, K_stride_d,
                      V_stride_N, V_stride_d,
                      OUT_stride_M, OUT_stride_d,
                      BLOCKSIZE_M: tl.constexpr,
                      BLOCKSIZE_N: tl.constexpr,
                      BLOCKSIZE_d: tl.constexpr):
    
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    offset_M = pid0 * BLOCKSIZE_M + tl.arange(0, BLOCKSIZE_M)
    offset_d = pid1 * BLOCKSIZE_d + tl.arange(0, BLOCKSIZE_d)
    offset_N = tl.arange(0, BLOCKSIZE_N)

    mask_M = offset_M < M
    mask_d = offset_d < d

    accumulator = tl.zeros((BLOCKSIZE_M, BLOCKSIZE_d), dtype=tl.float32)
    softmax_running_sum = tl.zeros([BLOCKSIZE_M], dtype=tl.float32)
    softmax_current_max = tl.full([BLOCKSIZE_M], float("-inf"), dtype=tl.float32)
    attention_logits_scale = 1 / tl.sqrt(d + 0.0)

    for current_index in range(0, N, BLOCKSIZE_N):
        current_k_offset = current_index + offset_N
        current_v_offset = current_k_offset
        current_k_mask = current_k_offset < N

        complete_attention_logits = tl.zeros((BLOCKSIZE_M, BLOCKSIZE_N), dtype=tl.float32)

        for d_start in range(0, d, BLOCKSIZE_d):
            d_offset = d_start + tl.arange(0, BLOCKSIZE_d)
            d_mask = d_offset < d

            Q_offset = offset_M[:, None] * Q_stride_M + d_offset[None, :] * Q_stride_d  
            Q_mask = mask_M[:, None] & d_mask[None, :]
            Q_slice = tl.load(Q_ptr + Q_offset, mask=Q_mask)

            K_offset = d_offset[:, None] * K_stride_d + current_k_offset[None, :] * K_stride_N
            K_mask = d_mask[:, None] & current_k_mask[None, :]
            K_slice = tl.load(K_ptr + K_offset, mask=K_mask)

            complete_attention_logits += tl.dot(Q_slice, K_slice)

        complete_attention_logits = complete_attention_logits * attention_logits_scale

        attention_mask = mask_M[:, None] & current_k_mask[None, :]
        complete_attention_logits = tl.where(attention_mask, complete_attention_logits, float("-inf"))

        current_block_max = tl.max(complete_attention_logits, axis=-1)
        max_value = tl.maximum(current_block_max, softmax_current_max)

        alpha = tl.exp(softmax_current_max - max_value)
        softmax_current_max = max_value

        attention_logits_shift = complete_attention_logits - max_value[:, None]
        softmax_weights = tl.exp(attention_logits_shift)
        softmax_denom = tl.sum(softmax_weights, axis=1)

        softmax_running_sum = tl.fma(softmax_running_sum, alpha, softmax_denom)

        V_offset = current_k_offset[:, None] * V_stride_N + offset_d[None, :] * V_stride_d
        V_mask = current_k_mask[:, None] & mask_d[None, :]
        V_slice = tl.load(V_ptr + V_offset, mask=V_mask)

        weighted_values = tl.dot(softmax_weights, V_slice)
        accumulator = tl.fma(accumulator, alpha[:, None], weighted_values)

    accumulator /= softmax_running_sum[:, None]
    
    OUT_offset = offset_M[:, None] * OUT_stride_M + offset_d[None, :] * OUT_stride_d
    OUT_mask = (offset_M[:, None] < M) & (offset_d[None, :] < d)

    tl.store(OUT_ptr + OUT_offset, accumulator.to(tl.float16), mask=OUT_mask)


# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int):    
    BLOCKSIZE_M = 16
    BLOCKSIZE_d = 128
    BLOCKSIZE_N = 64
    
    grid = (triton.cdiv(M, BLOCKSIZE_M), triton.cdiv(d, BLOCKSIZE_d))
    
    Q_stride_M, Q_stride_d = Q.stride()
    K_stride_N, K_stride_d = K.stride()
    V_stride_N, V_stride_d = V.stride()
    OUT_stride_M, OUT_stride_d = output.stride()
    
    softmax_attention[grid](
        Q_ptr=Q,
        K_ptr=K, 
        V_ptr=V,
        OUT_ptr=output,
        
        M=M,
        N=N,
        d=d,
        
        Q_stride_M=Q_stride_M,
        Q_stride_d=Q_stride_d,
        K_stride_N=K_stride_N,
        K_stride_d=K_stride_d,
        V_stride_N=V_stride_N,
        V_stride_d=V_stride_d,

        OUT_stride_M=OUT_stride_M,
        OUT_stride_d=OUT_stride_d,
        
        BLOCKSIZE_M=BLOCKSIZE_M,
        BLOCKSIZE_d=BLOCKSIZE_d,
        BLOCKSIZE_N=BLOCKSIZE_N,
        num_warps=4
    )