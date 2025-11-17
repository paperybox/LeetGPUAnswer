# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl
import math
@triton.jit
def stage_1_batch(
    input_ptr,
    step,  # 0
    n,
    BLOCK_SIZE: tl.constexpr,
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)

    pid_offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    stride = 1 << (step + 1)
    stride_off = 1 << step  #

    block_start = stride * tl.floor(pid_offset / stride_off)  #

    pid_off = pid_offset % stride_off
    off_x = block_start + pid_off  # 3
    off_y = block_start + stride - 1 - pid_off  # 4

    off_x = off_x.to(tl.int32)
    off_y = off_y.to(tl.int32)

    x = tl.load(input_ptr + off_x, mask=off_x < n, other=float("inf"))
    y = tl.load(input_ptr + off_y, mask=off_y < n, other=float("inf"))
    write_msk = y < x
    tl.store(input_ptr + off_x, y, mask=(off_x < n) & write_msk)
    tl.store(input_ptr + off_y, x, mask=(off_y < n) & write_msk)



@triton.jit
def stage_2_batch(input_ptr, step, n, BLOCK_SIZE: tl.constexpr):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    pid_offset = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE

    stride = 1 << (step + 1)
    stride_off = 1 << step  #

    off_x = stride * tl.floor(pid_offset / stride_off) + pid_offset % stride_off
    off_y = off_x + stride_off

    off_x = off_x.to(tl.int32)
    off_y = off_y.to(tl.int32)

    x = tl.load(input_ptr + off_x, mask=off_x < n, other=float("inf"))
    y = tl.load(input_ptr + off_y, mask=off_y < n, other=float("inf"))
    write_msk = y < x
    tl.store(input_ptr + off_x, y, mask=(off_x < n) & write_msk)
    tl.store(input_ptr + off_y, x, mask=(off_y < n) & write_msk)

# data_ptr is a raw device pointer
def solve(data_ptr: int, N: int):
    BLOCK_SIZE = 1024
    # n_loop = torch.ceil(torch.log2(torch.tensor(n))).int().item()
    n_pow2 = triton.next_power_of_2(N)
    n_loop = int(math.log2(n_pow2))
    grid2 = (triton.cdiv((2**n_loop) // 2, BLOCK_SIZE),)
    for i in range(n_loop):
        stage_1_batch[grid2](data_ptr, i, N, BLOCK_SIZE=BLOCK_SIZE)
        for j in range(i):
            stage_2_batch[grid2](data_ptr, i - j - 1, N, BLOCK_SIZE=BLOCK_SIZE)


data = [5.0, 2.0, 8.0, 1.0, 9.0, 4.0]
data = torch.tensor(data=data,device="cuda",dtype=torch.float32)
N = data.shape[0]

print(data)
solve(data=data,N=N)
print(data)