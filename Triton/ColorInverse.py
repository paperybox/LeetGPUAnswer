import torch
import triton
import triton.language as tl

@triton.jit
def invert_kernel(
    image_ptr,
    width, height,
    BLOCK_SIZE: tl.constexpr
):
    image_ptr = image_ptr.to(tl.pointer_type(tl.uint32))
    img_sz = width * height
    pid = tl.program_id(axis=0)
    
    
    offset = BLOCK_SIZE * pid + tl.arange(0,BLOCK_SIZE)
    mask = offset < img_sz
    p = tl.load(image_ptr + offset,mask=mask,other=0)
    new_p = (p & 0xFF000000) | (~p & 0x00FFFFFF)    
    tl.store(image_ptr + offset,new_p,mask=mask)
    



    pass

# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    
    invert_kernel[grid](
        image,
        width, height,
        BLOCK_SIZE
    ) 



# Input: image = [255, 0, 128, 255, 0, 255, 0, 255], width=1, height=2
# Output: [0, 255, 127, 255, 255, 0, 255, 255]

image = torch.tensor([255, 0, 128, 255,
                       0, 255, 0, 255],dtype=torch.uint8,device='cuda')
width = 1
height = 2

# output = torch.zeros_like(image,dtype=torch.uint8,device='cuda')
print(f"image:{image}")

solve(image=image,width=width,height=height)
print(f"image:{image}")
