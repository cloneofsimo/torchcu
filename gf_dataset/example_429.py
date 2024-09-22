
import torch

def torch_full_like_fp32_function(input_tensor: torch.Tensor, value: float) -> torch.Tensor:
    """
    Creates a tensor of the same size and shape as the input tensor, filled with the specified value. 
    Returns a tensor of float32 dtype.
    """
    output_tensor = torch.full_like(input_tensor, value, dtype=torch.float32)
    return output_tensor

function_signature = {
    "name": "torch_full_like_fp32_function",
    "inputs": [
        ((4, 4), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
