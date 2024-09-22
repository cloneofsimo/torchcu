
import torch
from cutlass import *

def fused_linear_relu_bfloat16(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a fused linear transformation, bias addition, and ReLU activation using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)
    output = torch.nn.functional.linear(input_bf16, weight_bf16, bias_bf16)
    return torch.relu(output).to(torch.float32)

function_signature = {
    "name": "fused_linear_relu_bfloat16",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
