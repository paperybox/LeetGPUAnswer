import torch
import triton
import triton.language as tl


@triton.jit
def solve_kernel(input_ptr: torch.Tensor, output_ptr: torch.Tensor, N: int, 
                 BLOCK_SIZE: tl.constexpr):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    off = BLOCK_SIZE * pid + tl.arange(0,BLOCK_SIZE)
    mask = off < N
    input = tl.load(input_ptr + off, mask=mask)
    res = tl.sum(input.to(tl.float64))
    tl.atomic_add(output_ptr,res)

    pass

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = triton.cdiv(N,BLOCK_SIZE)
    solve_kernel[(grid,)](input,output,N,BLOCK_SIZE)



# Input: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
# Output: 36.0

input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
input = torch.tensor(input,dtype=torch.float32,device='cuda')
output = torch.tensor([0],dtype=torch.float32,device='cuda')


solve(input,output,len(input))


print(output)


