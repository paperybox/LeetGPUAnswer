import torch
import triton
import triton.language as tl

@triton.jit
def reverse_kernel(
    input_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    off = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = off < N
    input = tl.load(input_ptr + off, mask=mask,other=0)
    # tl.device_print("input", input)
    w_off = N - 1 - off
    w_mask = w_off >= 0

    tl.store(input_ptr + w_off, input,mask=w_mask)


    pass

# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(N // 2, BLOCK_SIZE)
    grid = (n_blocks,)
    
    reverse_kernel[grid](
        input,
        N,
        BLOCK_SIZE
    ) 

Input= torch.tensor([1.0, 2.0, 3.0, 4.0],dtype=torch.float32,device='cuda')
# Output= [4.0, 3.0, 2.0, 1.0]
print(Input)

solve(Input,len(Input))

print(Input)
