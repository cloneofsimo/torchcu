
import torch

def torch_reflect_pad2d_function(input_tensor: torch.Tensor, padding: int) -> torch.Tensor:
    """
    Performs reflection padding on a 2D tensor.
    """
    return torch.nn.functional.pad(input_tensor, (padding, padding, padding, padding), mode='reflect')

function_signature = {
    "name": "torch_reflect_pad2d_function",
    "inputs": [
        ((2, 3, 4, 5), torch.float32),
        (1, torch.int32)
    ],
    "outputs": [
        ((2, 3, 4 + 2 * 1, 5 + 2 * 1), torch.float32)
    ]
}
