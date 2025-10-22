import torch
import triton
import numpy as np





def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int):
    val = torch.matmul(Q,K.T) / d**0.5
    val = torch.softmax(val,dim=-1)
    torch.matmul(val,V,out=output)