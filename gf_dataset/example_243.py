
import torch
import torch.nn.functional as F
from cutlass import *

def torch_pow_preactivation_bf16(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Applies a power function, pre-activation, and elementwise difference.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    output = torch.pow(input_bf16, 2.0)  # Apply power function
    output = torch.matmul(output, weight_bf16.t()) + bias_bf16 # Pre-activation
    output = torch.sigmoid(output)  # Apply sigmoid activation

    # Elementwise difference between input and output
    elementwise_diff = torch.abs(input_bf16 - output)

    return elementwise_diff.to(torch.float32)

function_signature = {
    "name": "torch_pow_preactivation_bf16",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 1), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
