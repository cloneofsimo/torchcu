
import torch
import torch.nn.functional as F

def torch_conv2d_relu_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a 2D convolution with ReLU activation using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    output = F.conv2d(input_bf16, weight_bf16, bias_bf16, stride=1, padding=1)
    output = F.relu(output)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_conv2d_relu_function",
    "inputs": [
        ((1, 1, 4, 4), torch.float32),  # Input shape (batch, channels, height, width)
        ((1, 1, 3, 3), torch.float32),  # Weight shape (out_channels, in_channels, kernel_height, kernel_width)
        ((1,), torch.float32)           # Bias shape (out_channels)
    ],
    "outputs": [
        ((1, 1, 4, 4), torch.float32),  # Output shape (batch, out_channels, height, width)
    ]
}
