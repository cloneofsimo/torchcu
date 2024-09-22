
import torch
import torch.nn.functional as F

def reflection_pad_function(input_tensor: torch.Tensor, padding: int) -> torch.Tensor:
    """
    Applies reflection padding to the input tensor.
    """
    return F.pad(input_tensor, (padding, padding, padding, padding), mode='reflect')

function_signature = {
    "name": "reflection_pad_function",
    "inputs": [
        ((1, 1, 3, 3), torch.float32),
        (1, torch.int32)
    ],
    "outputs": [
        ((1, 1, 3 + 2 * 1, 3 + 2 * 1), torch.float32)
    ]
}
