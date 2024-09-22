
import torch

def torch_noisy_relu_inplace(input_tensor: torch.Tensor, noise_scale: float) -> torch.Tensor:
    """
    Applies ReLU activation with inplace noise injection.
    """
    noise = torch.randn_like(input_tensor) * noise_scale
    input_tensor.add_(noise)
    input_tensor.relu_()
    return input_tensor

function_signature = {
    "name": "torch_noisy_relu_inplace",
    "inputs": [
        ((4, 4), torch.float32),
        ((), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
