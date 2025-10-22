import torch
import triton
import triton.language as tl

# a, b are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    b.copy_(a) 



A = [[1.0, 2.0],
             [3.0, 4.0]]
A = torch.tensor(A,dtype=torch.float32,device='cuda')
B = torch.zeros_like(A,dtype=torch.float32,device='cuda')

solve(A,B,len(A))

print(B)

