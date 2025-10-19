import torch
import triton
import triton.language as tl

@triton.jit
def fnv1a_hash(x):
    FNV_PRIME = 16777619
    OFFSET_BASIS = 2166136261
    
    hash_val = tl.full(x.shape, OFFSET_BASIS, tl.uint32)
    
    for byte_pos in range(4):
        byte = (x >> (byte_pos * 8)) & 0xFF
        hash_val = (hash_val ^ byte) * FNV_PRIME
    
    return hash_val

@triton.jit
def fnv1a_hash_kernel(
    input,
    output,
    n_elements,
    n_rounds,
    BLOCK_SIZE: tl.constexpr
):
    input = input.to(tl.pointer_type(tl.uint32))

    pid = tl.program_id(0)

    off = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = off < n_elements

    num = tl.load(input + off, mask=mask)

    # out = tl.zeros_like(num)
    for i in range(0,n_rounds):
        num = fnv1a_hash(num)
    tl.store(output+off,num,mask=mask)



# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, R: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    fnv1a_hash_kernel[grid](
        input,
        output,
        N,
        R,
        BLOCK_SIZE
    )


numbers = [123, 456, 789]
R = 2

numbers = torch.tensor(numbers,dtype=torch.uint32,device='cuda')
output = torch.zeros_like(numbers,dtype=torch.uint32,device='cuda')
solve(numbers,output,len(numbers),R)
print(output)

# Output: hashes = [1636807824, 1273011621, 2193987222]