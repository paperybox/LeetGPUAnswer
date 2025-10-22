import torch
import triton
import triton.language as tl

@triton.jit
def reduce_max_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    tl.atomic_max(output_ptr, tl.max(values, axis=0))

@triton.jit
def softmax_kernel_sum(
    input_ptr,
    N,
    sum_ptr, 
    max_ptr,
    BS: tl.constexpr
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    sum_ptr = sum_ptr.to(tl.pointer_type(tl.float32))
    max_ptr = max_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    off = BS * pid + tl.arange(0,BS)
    mask = off < N
    max = tl.load(max_ptr)

    val = tl.load(input_ptr+off,mask=mask,other=-float('inf'))
    s = tl.sum(tl.exp(val-max))
    tl.atomic_add(sum_ptr,s)

@triton.jit
def softmax_kernel_reduce(
    input_ptr, output_ptr,
    N,
    sum_ptr ,
    max_ptr ,
    BS: tl.constexpr
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    sum_ptr = sum_ptr.to(tl.pointer_type(tl.float32))
    max_ptr = max_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    off = BS * pid + tl.arange(0,BS)
    mask = off < N

    sum_val = tl.load(sum_ptr)
    max_val = tl.load(max_ptr)

    input = tl.load(input_ptr+off,mask=mask,other=-float('inf'))
    val = tl.exp(input - max_val) / sum_val

    tl.store(output_ptr+off,val,mask=mask)



# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BS = 256
    grid = (triton.cdiv(N,BS),)

    max_val = torch.tensor([-float('inf'),],dtype=torch.float32,device='cuda')
    sum_val = torch.zeros(1,dtype=torch.float32,device='cuda')

    reduce_max_kernel[grid](input,max_val,N,BS)
    softmax_kernel_sum[grid](input,N,sum_val,max_val,BS)
    softmax_kernel_reduce[grid](input,output,N,sum_val,max_val,BS)




Input= [10000,1.0, 2.0, 3.0]
N = len(Input)
Input = torch.tensor(Input,dtype=torch.float32,device='cuda')
output = torch.zeros_like(Input,dtype=torch.float32,device='cuda')

solve(Input,output,N)
print(output)
print(torch.softmax(Input,dim=-1))


