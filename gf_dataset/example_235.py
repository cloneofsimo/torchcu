
import torch

def torch_add_and_fill_function(input_tensor: torch.Tensor, value: float) -> torch.Tensor:
    """
    Adds a scalar value to an input tensor and fills empty elements with the value.
    """
    output = input_tensor + value
    output[output == 0] = value
    return output

function_signature = {
    "name": "torch_add_and_fill_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
