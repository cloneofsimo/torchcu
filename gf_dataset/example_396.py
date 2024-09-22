
import torch
import torch.nn.functional as F

def torch_conv2d_relu_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a 2D convolution with ReLU activation.
    """
    output = F.conv2d(input_tensor, weight, bias=bias)
    return F.relu(output)


function_signature = {
    "name": "torch_conv2d_relu_function",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),  # Input tensor shape (batch, channels, height, width)
        ((32, 3, 3, 3), torch.float32),    # Weight tensor shape (out_channels, in_channels, kernel_height, kernel_width)
        ((32,), torch.float32),           # Bias tensor shape (out_channels,)
    ],
    "outputs": [
        ((1, 32, 224, 224), torch.float32),  # Output tensor shape (batch, out_channels, height, width)
    ]
}
