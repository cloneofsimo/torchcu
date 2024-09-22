
import torch
import numpy as np

def roberts_cross_gradient_fp32(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies Roberts cross-gradient edge detection filter on the input tensor.
    """
    # Define the Roberts cross-gradient kernels
    kernel_x = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32)
    kernel_y = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32)

    # Apply convolution with the kernels
    gradient_x = torch.nn.functional.conv2d(input_tensor.unsqueeze(0).unsqueeze(0), kernel_x.unsqueeze(0).unsqueeze(0), padding=1)
    gradient_y = torch.nn.functional.conv2d(input_tensor.unsqueeze(0).unsqueeze(0), kernel_y.unsqueeze(0).unsqueeze(0), padding=1)

    # Calculate the magnitude of the gradient
    gradient_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)
    return gradient_magnitude.squeeze()

function_signature = {
    "name": "roberts_cross_gradient_fp32",
    "inputs": [
        ((128, 128), torch.float32)
    ],
    "outputs": [
        ((128, 128), torch.float32),
    ]
}
