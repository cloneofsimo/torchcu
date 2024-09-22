
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def outer_product_fp16_function(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor) -> torch.Tensor:
    """
    Computes the outer product of two tensors using FP16 and returns the result in FP16.
    """
    with autocast():
        output = torch.outer(input_tensor1, input_tensor2)
    return output.to(torch.float16)


function_signature = {
    "name": "outer_product_fp16_function",
    "inputs": [
        ((8,), torch.float32),
        ((16,), torch.float32)
    ],
    "outputs": [
        ((8, 16), torch.float16),
    ]
}
