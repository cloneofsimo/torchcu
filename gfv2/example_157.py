
import torch
import torch.nn as nn

def adaptive_max_pool1d_function(input_tensor: torch.Tensor, output_size: int) -> torch.Tensor:
    """
    Performs adaptive max pooling in 1D.
    """
    return nn.AdaptiveMaxPool1d(output_size=output_size)(input_tensor)

function_signature = {
    "name": "adaptive_max_pool1d_function",
    "inputs": [
        ((4, 4), torch.float32),
        (1, torch.int32)
    ],
    "outputs": [
        ((4, 1), torch.float32)
    ]
}

