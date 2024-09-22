
import torch

def tanh_bfloat16_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies the tanh activation function to the input tensor using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output = torch.tanh(input_bf16)
    return output.to(torch.float32)

function_signature = {
    "name": "tanh_bfloat16_function",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
