import torch
import triton
import triton.language as tl



@triton.jit
def step1_kernel(input_ptr, output_ptr, max_ptr, sum_ptr, length, BLOCK_SIZE: tl.constexpr):
    # Block Max
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    max_ptr = max_ptr.to(tl.pointer_type(tl.float32))
    sum_ptr = sum_ptr.to(tl.pointer_type(tl.float32))


    pid = tl.program_id(0)
    off = BLOCK_SIZE * pid +  tl.arange(0,BLOCK_SIZE)
    mask = off < length
    inputs = input_ptr + off

    val = tl.load(inputs,mask=mask,other=-float('inf'))
    block_max = tl.max(val)
    val = tl.exp(val - block_max)
    block_sum = tl.sum(val)
    tl.store(output_ptr + off, value=val, mask=mask)
    tl.store(max_ptr+pid,value=block_max)
    tl.store(sum_ptr+pid,value=block_sum)
    pass


@triton.jit
def step2_kernel(max_ptr, sum_ptr, block_len, BLOCK_SIZE: tl.constexpr):
    assert(block_len<BLOCK_SIZE)
    
    max_ptr = max_ptr.to(tl.pointer_type(tl.float32))
    sum_ptr = sum_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0) # PID = 0
    off = tl.arange(0,BLOCK_SIZE)
    mask = off < block_len
    maxS = tl.load(max_ptr + off,mask=mask,other=-float('inf'))
    sumS = tl.load(sum_ptr + off, mask=mask,other=0)
    fix_e = tl.exp(maxS - tl.max(maxS))
    sumS *= fix_e
    sum = tl.sum(sumS)

    tl.store(max_ptr + off,fix_e, mask=mask)
    tl.store(sum_ptr,sum)

    pass


@triton.jit
def step3_kernel(input_ptr, output_ptr, fix_e_ptr,sum_ptr, length, BLOCK_SIZE: tl.constexpr):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    fix_e_ptr = fix_e_ptr.to(tl.pointer_type(tl.float32))
    sum_ptr = sum_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    off = BLOCK_SIZE * pid +  tl.arange(0,BLOCK_SIZE)
    mask = off < length

    sum = tl.load(sum_ptr)
    fix_e = tl.load(fix_e_ptr + pid)
    val = tl.load(input_ptr + off,mask=mask,other=0)
    val = val * fix_e / sum
    tl.store(output_ptr + off,value=val,mask=mask)
    pass

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BS = 1024
    grid_size = triton.cdiv(N,BS)

    maxs_val = torch.zeros(grid_size ,dtype=torch.float32,device='cuda')
    sums_val = torch.zeros(grid_size ,dtype=torch.float32,device='cuda')
    step1_kernel[(grid_size,)](input,output,maxs_val,sums_val,N,BS)
    step2_kernel[(1,)](maxs_val,sums_val,grid_size,triton.next_power_of_2(grid_size))
    step3_kernel[(grid_size,)](input_ptr=output ,output_ptr=output,fix_e_ptr=maxs_val,sum_ptr=sums_val,length=N,BLOCK_SIZE=BS)






Input= [1,3,-2,3,1,4,5,0,1]
# Input= [10000,1.0, 2.0, 3.0]
# val = np.exp(Input - np.max(Input))
# np_sf = val / np.sum(val)


N = len(Input)
Input = torch.tensor(Input,dtype=torch.float32,device='cuda')
output = torch.zeros_like(Input,dtype=torch.float32,device='cuda')

solve(Input,output,N)
print(output)
print(torch.softmax(Input,dim=-1))
# print(np_sf)






