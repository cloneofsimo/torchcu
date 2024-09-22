
import torch

def torch_clip_function(input_tensor: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Clips the input tensor values to the range [min_val, max_val].
    """
    return torch.clamp(input_tensor, min_val, max_val)

function_signature = {
    "name": "torch_clip_function",
    "inputs": [
        ((4, 4), torch.float32),
        (float, torch.float32),
        (float, torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
