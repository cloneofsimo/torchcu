
import torch

def my_function(input_tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Performs a simple operation:
    1. Clips the input tensor to the range [0, 1].
    2. Checks if the clipped tensor is not equal to the threshold.
    3. Returns the result as a float32 tensor.
    """
    output = torch.clip(input_tensor, 0.0, 1.0)
    output = (output != threshold).to(torch.float32)
    return output

function_signature = {
    "name": "my_function",
    "inputs": [
        ((1, 1, 1, 1), torch.float32),
        (torch.float32, )
    ],
    "outputs": [
        ((1, 1, 1, 1), torch.float32),
    ]
}
