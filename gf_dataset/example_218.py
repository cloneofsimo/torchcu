
import torch
import torch.nn.functional as F

def torch_noisy_norm_any_function(input_tensor: torch.Tensor, noise_scale: float) -> torch.Tensor:
    """
    Applies noise to the input tensor, calculates the norm of each element, 
    and returns True if any element's norm is greater than 1.
    """
    noise = torch.randn_like(input_tensor) * noise_scale
    noisy_input = input_tensor + noise
    norms = torch.norm(noisy_input, dim=1)
    return torch.any(norms > 1)

function_signature = {
    "name": "torch_noisy_norm_any_function",
    "inputs": [
        ((10, 5), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((), torch.bool),
    ]
}
