
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

def my_complex_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on the input tensor:
    1. Slices the input tensor
    2. Standardizes the weight tensor
    3. Performs a linear transformation with the standardized weight
    4. Applies ReLU activation
    5. Applies a layer normalization
    6. Slices the output
    
    This function utilizes gradient checkpointing, bfloat16 precision, and various optimizations for efficiency.
    """
    # Slice the input tensor
    input_slice = input_tensor[:, :3, :, :]

    # Standardize the weight tensor
    weight_mean = weight.mean()
    weight_std = weight.std()
    weight_standardized = (weight - weight_mean) / weight_std

    # Linear transformation with standardized weight
    output = checkpoint(lambda x: nn.functional.linear(x, weight_standardized.t()), input_slice.to(torch.bfloat16)).to(torch.float32)

    # ReLU activation
    output = torch.relu(output)

    # Layer normalization
    output = nn.LayerNorm(output.shape[1:])(output)

    # Slice the output
    output_slice = output[:, :2, :, :]

    return output_slice

function_signature = {
    "name": "my_complex_function",
    "inputs": [
        ((4, 5, 7, 7), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 2, 7, 7), torch.float32),
    ]
}
