
import torch

def softshrink_fp32_function(input_tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Apply the soft-shrinkage activation function.
    """
    return torch.where(input_tensor.abs() > threshold, input_tensor - torch.sign(input_tensor) * threshold, torch.tensor(0.0, dtype=torch.float32))


function_signature = {
    "name": "softshrink_fp32_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
