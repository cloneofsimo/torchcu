
import torch
import torch.nn.functional as F

def grid_sampler_fp16(input: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    Performs grid sampling on an input tensor using the provided grid.
    The grid contains normalized coordinates, with values between -1 and 1.
    Returns the output tensor in fp16.
    """
    input_fp16 = input.to(torch.float16)
    grid_fp16 = grid.to(torch.float16)
    output_fp16 = F.grid_sample(input_fp16, grid_fp16, mode='bilinear', align_corners=False)
    return output_fp16

function_signature = {
    "name": "grid_sampler_fp16",
    "inputs": [
        ((1, 1, 10, 10), torch.float32),
        ((1, 1, 10, 2), torch.float32)
    ],
    "outputs": [
        ((1, 1, 10, 10), torch.float16),
    ]
}
