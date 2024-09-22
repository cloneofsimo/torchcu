
import torch

def my_function(input_tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Applies a series of operations on the input tensor:
    1. Exponentiates the input tensor.
    2. Clamps the result between 0 and 1.
    3. Multiplies the clamped result by the scale.
    4. Converts the result to bfloat16.
    5. Creates a diagonal matrix using the result and returns the diagonal matrix.
    """
    output = torch.exp(input_tensor).clamp(0, 1) * scale
    output = output.to(torch.bfloat16)
    output = torch.diagflat(output).to(torch.float32)
    return output

function_signature = {
    "name": "my_function",
    "inputs": [
        ((1,), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((1, 1), torch.float32)
    ]
}
