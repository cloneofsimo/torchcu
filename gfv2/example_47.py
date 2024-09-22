
import torch

def pixel_unshuffle_fp16_function(input_tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Pixel unshuffle operation with layer scaling in fp16.
    """
    input_fp16 = input_tensor.to(torch.float16)
    output = torch.pixel_unshuffle(input_fp16, downscale_factor=2)
    output = output * scale
    return output.to(torch.float32)

function_signature = {
    "name": "pixel_unshuffle_fp16_function",
    "inputs": [
        ((8, 64, 8, 8), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((8, 64, 16, 16), torch.float32),
    ]
}
