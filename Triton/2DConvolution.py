import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(input_ptr, 
                  kernel_ptr,
                  output_ptr,
                  input_rows: int, 
                  input_cols: int, 
                  kernel_rows: int, 
                  kernel_cols: int,
                  BS_R:tl.constexpr,
                  BS_C:tl.constexpr):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    kernel_ptr = kernel_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))


    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    acc = tl.zeros([BS_R,BS_C],dtype=tl.float32)

    for r in range(0,kernel_rows):
        off_r = pid_r * BS_R + r + tl.arange(0,BS_R)
        mask_r = off_r < (input_rows - kernel_rows + r + 1)
        for c in range(0,kernel_cols):
            off_c = pid_c * BS_C + c + tl.arange(0,BS_C)
            mask_c = off_c < (input_cols - kernel_cols + c + 1)
            # tl.device_print("off:",off_r[:,None]*input_cols+off_c[None,:])
            # tl.device_print("mask:",mask_r[:,None] & mask_c[None,:])
            k = tl.load(kernel_ptr + r * kernel_cols + c)
            mat = tl.load(input_ptr+off_r[:,None]*input_cols+off_c[None,:],mask=mask_r[:,None] & mask_c[None,:],other=0)
            acc += mat * k
    
    out_off_r = pid_r * BS_R + tl.arange(0,BS_R)
    out_off_c = pid_c * BS_C + tl.arange(0,BS_C)

    mask_r = out_off_r < (input_rows - kernel_rows + 1)
    mask_c = out_off_c < (input_cols - kernel_cols + 1)
    tl.store(output_ptr + out_off_r[:,None] * (input_cols - kernel_cols + 1) + out_off_c[None,:],acc,mask=mask_r[:,None] & mask_c[None,:])
    pass


# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_rows: int, input_cols: int, kernel_rows: int, kernel_cols: int):
    BS_R=32
    BS_C=64
    grid = (triton.cdiv(input_rows,BS_R),triton.cdiv(input_cols,BS_C))

    conv2d_kernel[grid](input,kernel,output,input_rows,input_cols,kernel_rows,kernel_cols,BS_R,BS_C)


input = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
kernel = [[0, 1], [1, 0]]
input = torch.tensor(input, device="cuda", dtype=torch.float32)
kernel = torch.tensor(kernel, device="cuda", dtype=torch.float32)
output = torch.zeros(
    (input.shape[0] - kernel.shape[0] + 1, input.shape[1] - kernel.shape[1] + 1),
    device="cuda",
    dtype=torch.float32,
)
solve(
    input=input,
    kernel=kernel,
    output=output,
    input_rows=input.shape[0],
    input_cols=input.shape[1],
    kernel_rows=kernel.shape[0],
    kernel_cols=kernel.shape[1],
)
print(output)
