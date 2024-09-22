
import torch

def gradient_clipping_function(input_tensor: torch.Tensor, clip_value: float) -> torch.Tensor:
    """
    Clips the gradient of the input tensor in-place using the given value.
    """
    torch.nn.utils.clip_grad_value_(input_tensor, clip_value)
    return input_tensor

function_signature = {
    "name": "gradient_clipping_function",
    "inputs": [
        ((1,), torch.float32),
        (float, )
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
