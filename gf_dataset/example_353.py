
import torch
import torch.nn.functional as F

def gradient_magnitude_bf16(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the magnitude of the gradient of the input tensor using bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    grad_x = torch.abs(F.pad(input_bf16, (0, 1, 0, 0), 'constant', 0) - input_bf16)
    grad_y = torch.abs(F.pad(input_bf16, (0, 0, 0, 1), 'constant', 0) - input_bf16)
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2).to(torch.float32)
    return magnitude

function_signature = {
    "name": "gradient_magnitude_bf16",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
