
import torch
import torch.nn as nn
import torch.nn.functional as F

def complex_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a sequence of operations:
        1. Adaptive Average Pooling 1D
        2. Transposed Convolution 1D (with bias)
        3. Gradient Penalty (using bfloat16 for efficiency)
        4. ReLU Activation
    Returns the final output tensor.
    """
    # 1. Adaptive Average Pooling 1D
    pooled = F.adaptive_avg_pool1d(input_tensor, output_size=1)

    # 2. Transposed Convolution 1D
    conv_output = F.conv_transpose1d(pooled, weight, bias=bias)

    # 3. Gradient Penalty (using bfloat16)
    conv_output_bf16 = conv_output.to(torch.bfloat16)
    grad_output = torch.ones_like(conv_output_bf16, dtype=torch.bfloat16)
    grad_input = torch.autograd.grad(outputs=conv_output_bf16, inputs=input_tensor, grad_outputs=grad_output,
                                       create_graph=True, retain_graph=True)[0]
    grad_penalty = torch.mean(torch.square(grad_input.to(torch.float32)))

    # 4. ReLU Activation
    final_output = F.relu(conv_output)

    return final_output

function_signature = {
    "name": "complex_function",
    "inputs": [
        ((1, 4, 8), torch.float32),  # Input tensor
        ((4, 4, 3), torch.float32),  # Weight tensor
        ((4,), torch.float32)        # Bias tensor
    ],
    "outputs": [
        ((1, 4, 11), torch.float32)   # Output tensor
    ]
}
