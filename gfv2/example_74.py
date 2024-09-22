
import torch
import torch.nn.functional as F
from torch.nn import init

def sobel_gradient_int8_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Sobel gradient of an image using int8 precision.
    Returns the gradient as a single tensor with int8 dtype.
    """
    # Convert to int8
    input_tensor = input_tensor.to(torch.int8)
    # Calculate gradients using Sobel kernels
    sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.int8)
    sobel_y_kernel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.int8)
    grad_x = F.conv2d(input_tensor, sobel_x_kernel.unsqueeze(0).unsqueeze(0), padding=1)
    grad_y = F.conv2d(input_tensor, sobel_y_kernel.unsqueeze(0).unsqueeze(0), padding=1)
    # Combine gradients into a single tensor
    gradient = torch.cat([grad_x, grad_y], dim=1)
    return gradient.to(torch.int8)


function_signature = {
    "name": "sobel_gradient_int8_function",
    "inputs": [
        ((1, 1, 10, 10), torch.float32)
    ],
    "outputs": [
        ((1, 2, 10, 10), torch.int8),
    ]
}
