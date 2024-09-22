
import torch
import torch.nn as nn

def torch_pixel_shuffle_hardsigmoid_function(input_tensor: torch.Tensor, upscale_factor: int) -> torch.Tensor:
    """
    Performs pixel shuffle and hardsigmoid activation.
    """
    output = nn.PixelShuffle(upscale_factor)(input_tensor)
    output = torch.hardsigmoid(output)
    return output

function_signature = {
    "name": "torch_pixel_shuffle_hardsigmoid_function",
    "inputs": [
        ((16, 16, 4, 4), torch.float32),
        (1, torch.int32)
    ],
    "outputs": [
        ((16, 16, 16, 16), torch.float32),
    ]
}
