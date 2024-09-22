
import torch
import torch.nn as nn

def power_adaptive_max_pool(input_tensor: torch.Tensor, exponent: float, output_size: int) -> torch.Tensor:
    """
    Applies element-wise power operation, then adaptive max pooling 2D.
    """
    input_tensor.pow_(exponent)  # In-place power operation
    output = nn.AdaptiveMaxPool2d(output_size)(input_tensor)
    return output

function_signature = {
    "name": "power_adaptive_max_pool",
    "inputs": [
        ((1, 3, 10, 10), torch.float32),
        (float, ),
        (int, )
    ],
    "outputs": [
        ((1, 3, output_size, output_size), torch.float32),
    ]
}
