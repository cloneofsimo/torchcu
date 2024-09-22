
import torch
from torch.nn.functional import conv2d
import numpy as np

def torch_image_edge_enhancement(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Enhances edges in an image using Scharr gradient and Laplacian filtering, then applies a threshold.
    """
    # Scharr Gradient
    grad_x = conv2d(image, kernel, padding='same')
    grad_y = conv2d(image, kernel.transpose(2, 3), padding='same')
    gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    # Laplacian
    laplacian_kernel = torch.tensor([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    laplacian = conv2d(image, laplacian_kernel, padding='same')

    # Combine and threshold
    enhanced_image = gradient_magnitude + laplacian
    enhanced_image = torch.where(enhanced_image > 0.2, enhanced_image, 0.0)  # Threshold

    return enhanced_image

function_signature = {
    "name": "torch_image_edge_enhancement",
    "inputs": [
        ((1, 1, 32, 32), torch.float32),  # Image (batch, channels, height, width)
        ((1, 1, 3, 3), torch.float32),  # Scharr kernel
    ],
    "outputs": [
        ((1, 1, 32, 32), torch.float32),  # Enhanced image
    ]
}
