
import torch

def torch_logspace_add(input_tensor: torch.Tensor, start: float, end: float, steps: int) -> torch.Tensor:
    """
    Generate a logarithmically spaced tensor and add it to the input.
    """
    logspace_tensor = torch.logspace(start, end, steps)
    return input_tensor + logspace_tensor.to(input_tensor.dtype)

function_signature = {
    "name": "torch_logspace_add",
    "inputs": [
        ((4, 4), torch.float32),
        (torch.float32),
        (torch.float32),
        (torch.int32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
