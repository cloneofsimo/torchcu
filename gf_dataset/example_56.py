
import torch

def torch_square_bfloat16_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate the square of a tensor using bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output = torch.square(input_bf16)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_square_bfloat16_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
