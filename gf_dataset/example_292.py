
import torch
import torch.nn.functional as F

def sobel_gradient_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the Sobel gradient of an image using bfloat16 for memory efficiency.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    
    # Sobel kernel for x-direction
    sobel_x_kernel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.bfloat16)
    
    # Sobel kernel for y-direction
    sobel_y_kernel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.bfloat16)
    
    # Apply convolution
    grad_x = F.conv2d(input_bf16, sobel_x_kernel.unsqueeze(0).unsqueeze(0), padding=1)
    grad_y = F.conv2d(input_bf16, sobel_y_kernel.unsqueeze(0).unsqueeze(0), padding=1)
    
    # Combine gradients
    gradient = torch.sqrt(grad_x**2 + grad_y**2).to(torch.float32)
    
    return gradient

function_signature = {
    "name": "sobel_gradient_function",
    "inputs": [
        ((1, 1, 10, 10), torch.float32)  # Assuming a single channel image with size 10x10
    ],
    "outputs": [
        ((1, 1, 10, 10), torch.float32),
    ]
}

