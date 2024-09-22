
import torch
import torch.nn.functional as F
from cutlass import *

def fading_out_outer_product_int8(input_tensor: torch.Tensor, weight: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    """
    Computes the outer product of two tensors, applies a fading-out function, and quantizes to int8.

    Args:
        input_tensor: Input tensor of shape (batch_size, input_dim).
        weight: Weight tensor of shape (output_dim, input_dim).
        scaling_factor: Scaling factor for the fading-out function.

    Returns:
        A tensor of shape (batch_size, output_dim) containing the quantized results.
    """
    # Apply fading-out function
    fading_out = torch.exp(-torch.arange(input_tensor.size(1), device=input_tensor.device) * scaling_factor)
    input_tensor = input_tensor * fading_out

    # Compute outer product
    output = torch.einsum("bi,oj->bo", input_tensor, weight)

    # Quantize to int8
    output_int8 = torch.quantize_per_tensor(output, scale=1.0, zero_point=0, dtype=torch.qint8)

    return output_int8


function_signature = {
    "name": "fading_out_outer_product_int8",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.qint8),
    ]
}
