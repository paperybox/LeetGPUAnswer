import torch
import triton
import triton.language as tl

import numpy as np

@triton.jit
def matrix_multiplication_kernel(
    a, b, c, 
    M, N, K,
    BS_M: tl.constexpr,
    BS_N: tl.constexpr,
    BS_K: tl.constexpr
):
    a = a.to(tl.pointer_type(tl.float32))
    b = b.to(tl.pointer_type(tl.float32))
    c = c.to(tl.pointer_type(tl.float32))

    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    off_m = (pid_m * BS_M + tl.arange(0,BS_M)) 
    off_k = (pid_k * BS_K + tl.arange(0,BS_K))

    STEPS = tl.cdiv(N,BS_N)

    res = tl.zeros([BS_M,BS_K],tl.float32)
    for step in range(0,STEPS):
        off_n = step * BS_N + tl.arange(0,BS_N)
        
        mat_a = tl.load(a + (off_m*N)[:,None] + (off_n)[None,:],mask=(off_m<M)[:,None] * (off_n<N)[None,:], other=0)
        mat_b = tl.load(b + (off_n*K)[:,None] + off_k[None,:],mask=(off_n<N)[:,None] * (off_k<K)[None,:], other=0)
        res += tl.dot(mat_a,mat_b)   
    tl.store(c+((off_m*K)[:,None]+off_k[None,:]) ,res,mask=(off_m<M)[:,None]*(off_k<K)[None,:])


    pass

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    BS_M,BS_N,BS_K = 64,64,64
    
    grid = (triton.cdiv(M,BS_M), triton.cdiv(K,BS_K)) 
    matrix_multiplication_kernel[grid](
        a, b, c,
        M, N, K,
        BS_M,BS_N,BS_K
    )


A = [[-2.356018543243408, 6.935468673706055, -7.013333320617676, 6.163064002990723], [-3.5692100524902344, -1.6004270315170288, -8.937905311584473, -7.993679523468018], [-6.5450520515441895, -7.334951877593994, -4.342578887939453, -4.810895919799805], [7.715483665466309, 6.502724647521973, -7.561675071716309, -4.331381320953369]]
B = [[-6.381594657897949, 4.148236274719238, 7.6836957931518555, -5.287285327911377], [-9.508222579956055, 8.838855743408203, -8.600454330444336, -2.4791955947875977], [7.677063941955566, 8.191507339477539, -4.250603675842285, -6.241643905639648], [-6.623974323272705, -5.892275810241699, 3.9410316944122314, -4.3462347984313965]]

matrix1 = torch.tensor(A, dtype=torch.float32, device="cuda")
matrix2 = torch.tensor(B, dtype=torch.float32, device="cuda")
out = torch.zeros_like(matrix1, dtype=torch.float32, device="cuda")
# print(len(matrix1),len(matrix1[0]),len(matrix2[0]))

solve(matrix1, matrix2, out, len(matrix1), len(matrix1[0]), len(matrix2[0]))

torch_res = torch.matmul(matrix1, matrix2)


print(torch_res-out)


import torch.nn.functional as F
num = F.cosine_similarity(torch_res, out,1)
print("{}".format(num))



# Expected= [[-145.57460021972656, -42.2359619140625, -23.651386260986328, 12.251165390014648], [22.327529907226562, -55.06583023071289, -7.172174453735352, 113.36883544921875], [110.03921508789062, -99.20814514160156, 12.29241943359375, 100.80445861816406], [-140.42694091796875, 53.062477111816406, 18.42861557006836, 9.106996536254883]]
# Got= [[-145.50538635253906, -42.19313049316406, -23.61658477783203, 12.246894836425781], [22.285743713378906, -55.02002716064453, -7.171653747558594, 113.29469299316406], [109.97502136230469, -99.10975646972656, 12.237236022949219, 100.71366119384766], [-140.37423706054688, 53.01737976074219, 18.491928100585938, 9.06842041015625]]
 
 # Max abs diff: 0.098388671875
