
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

@custom_fwd(cast_inputs=torch.bfloat16)
def torch_sobel_bfloat16_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate Sobel gradients using bfloat16 for efficiency.
    """
    # Assume input_tensor is a 4D tensor: (batch, channels, height, width)
    
    # Sobel kernels
    sobel_x_kernel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.bfloat16)
    sobel_y_kernel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.bfloat16)
    
    # Calculate gradients along x and y
    grad_x = F.conv2d(input_tensor, sobel_x_kernel.unsqueeze(0).unsqueeze(0), padding=1)
    grad_y = F.conv2d(input_tensor, sobel_y_kernel.unsqueeze(0).unsqueeze(0), padding=1)
    
    return torch.stack([grad_x, grad_y], dim=1)

function_signature = {
    "name": "torch_sobel_bfloat16_function",
    "inputs": [
        ((1, 1, 10, 10), torch.float32),
    ],
    "outputs": [
        ((1, 2, 10, 10), torch.float32)
    ]
}
