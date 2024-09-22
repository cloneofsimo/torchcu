
import torch
import torch.nn.functional as F

def sobel_adaptive_max_pool(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes Sobel gradients and applies adaptive max pooling.
    """
    # Calculate Sobel gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
    grad_x = F.conv2d(input_tensor, sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
    grad_y = F.conv2d(input_tensor, sobel_y.unsqueeze(0).unsqueeze(0), padding=1)

    # Combine gradient magnitudes
    gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    # Apply adaptive max pooling
    output = F.adaptive_max_pool2d(gradient_magnitude, (1, 1))
    
    return output

function_signature = {
    "name": "sobel_adaptive_max_pool",
    "inputs": [
        ((1, 1, 32, 32), torch.float32)
    ],
    "outputs": [
        ((1, 1, 1, 1), torch.float32)
    ]
}
