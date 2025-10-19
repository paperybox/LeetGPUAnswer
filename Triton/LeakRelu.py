import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    off = BLOCK_SIZE * pid + tl.arange(0,BLOCK_SIZE)
    mask = off < n_elements
    input = tl.load(input_ptr + off,mask=mask).to(tl.float32)
    output = max(0,input)
    tl.store(output_ptr+off,output,mask=mask)

    
    



# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_kernel[grid](input, output, N, BLOCK_SIZE)


# input = [-2.0, -1.0, 0.0, 1.0, 2.0]
# Output: output = [0.0, 0.0, 0.0, 1.0, 2.0]

input = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0],dtype=torch.float32,device='cuda')
ouput = torch.zeros_like(input,dtype=torch.float32,device='cuda')
print(input)
solve(input,ouput,len(input))
print(ouput)
