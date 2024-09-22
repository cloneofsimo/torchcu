
import torch

def ones_bf16_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Creates a tensor filled with ones of the same size as the input tensor and returns it in bfloat16.
    """
    ones_tensor = torch.ones_like(input_tensor, dtype=torch.bfloat16)
    return ones_tensor

function_signature = {
    "name": "ones_bfloat16_function",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.bfloat16),
    ]
}
