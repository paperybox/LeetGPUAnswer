import torch
import triton
import triton.language as tl

@triton.jit
def count_equal_kernel(input_ptr, output_ptr, N, K, BLOCK_SIZE: tl.constexpr):
    input_ptr = input_ptr.to(tl.pointer_type(tl.int32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.int32))

    pid = tl.program_id(0)
    off = BLOCK_SIZE * pid + tl.arange(0,BLOCK_SIZE)
    mask = off < N
    input = tl.load(input_ptr + off,mask=mask)
    res = tl.sum((input == K) & mask)
    tl.atomic_add(output_ptr,res)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, K: int):
    
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    count_equal_kernel[grid](input, output, N, K, BLOCK_SIZE=BLOCK_SIZE)


input = [1, 2, 3, 4, 1]
input = torch.tensor(input,dtype=torch.int32,device='cuda')
k = 1
output = torch.tensor([0],dtype=torch.int32,device='cuda')
print(output)

solve(input,output,len(input),k)
print(output)
