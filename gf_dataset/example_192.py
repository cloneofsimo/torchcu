
import torch
import torch.nn.functional as F
from cutlass import *

def sobel_gradient_cuda(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Sobel gradient of an image using CUDA.
    """
    # Define Sobel kernels
    sobel_x_kernel = torch.tensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]], dtype=torch.float32)
    sobel_y_kernel = torch.tensor([[ 1,  2,  1],
                                    [ 0,  0,  0],
                                    [-1, -2, -1]], dtype=torch.float32)

    # Perform 2D convolution with Sobel kernels
    grad_x = F.conv2d(input_tensor.unsqueeze(0).unsqueeze(0), sobel_x_kernel.unsqueeze(0).unsqueeze(0), padding=1)
    grad_y = F.conv2d(input_tensor.unsqueeze(0).unsqueeze(0), sobel_y_kernel.unsqueeze(0).unsqueeze(0), padding=1)

    # Combine gradients
    gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    return gradient.squeeze(0).squeeze(0)

function_signature = {
    "name": "sobel_gradient_cuda",
    "inputs": [
        ((8, 8), torch.float32)
    ],
    "outputs": [
        ((8, 8), torch.float32),
    ]
}
