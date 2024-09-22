
import torch
import torch.nn.functional as F

def torch_gradient_magnitude_tanh_bmm(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Calculates the gradient magnitude of the input tensor, applies tanh, 
    and performs a batched matrix multiplication with the weight tensor. 
    """
    # Gradient magnitude calculation
    grad_magnitude = torch.sqrt(torch.sum(torch.square(torch.abs(torch.grad(input_tensor.sum(), input_tensor))), dim=1))

    # Apply tanh
    tanh_output = F.tanh(grad_magnitude)

    # Batched matrix multiplication
    output = torch.bmm(tanh_output.unsqueeze(1), weight)

    return output

function_signature = {
    "name": "torch_gradient_magnitude_tanh_bmm",
    "inputs": [
        ((2, 3, 4), torch.float32),
        ((4, 5), torch.float32)
    ],
    "outputs": [
        ((2, 1, 5), torch.float32),
    ]
}
