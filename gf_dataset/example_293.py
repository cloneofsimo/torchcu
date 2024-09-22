
import torch
from torch.nn.functional import conv2d

def torch_sobel_erosion_fp32_function(input_tensor: torch.Tensor, kernel_size: int, erosion_kernel: torch.Tensor) -> torch.Tensor:
    """
    Applies a Sobel filter to an input tensor, then performs morphological erosion. 
    
    Args:
        input_tensor: The input tensor with shape (batch_size, channels, height, width)
        kernel_size: The size of the Sobel kernel (e.g., 3 for a 3x3 kernel)
        erosion_kernel: The kernel used for morphological erosion. Should be a 2D tensor.

    Returns:
        A tensor representing the result of the Sobel filter followed by erosion.
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).reshape(1, 1, 3, 3)

    # Apply Sobel filter
    grad_x = conv2d(input_tensor, sobel_x, padding=1)
    grad_y = conv2d(input_tensor, sobel_y, padding=1)
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

    # Apply morphological erosion
    eroded = torch.nn.functional.max_pool2d(gradient_magnitude, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2))

    return eroded

function_signature = {
    "name": "torch_sobel_erosion_fp32_function",
    "inputs": [
        ((1, 1, 10, 10), torch.float32),
        (3, torch.int32),
        ((3, 3), torch.float32)
    ],
    "outputs": [
        ((1, 1, 10, 10), torch.float32),
    ]
}
