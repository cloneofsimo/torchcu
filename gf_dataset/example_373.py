
import torch
import torch.nn.functional as F

def torch_scaled_softshrink_int8_function(input_tensor: torch.Tensor, scale: float, lambd: float) -> torch.Tensor:
    """
    Applies scaled softshrink activation to the input tensor and returns the result in int8.
    """
    output = F.softshrink(input_tensor * scale, lambd)
    return output.to(torch.int8)

function_signature = {
    "name": "torch_scaled_softshrink_int8_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((), torch.float32),
        ((), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.int8)
    ]
}
