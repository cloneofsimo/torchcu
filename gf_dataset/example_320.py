
import torch

def torch_hardshrink_bfloat16_function(input_tensor: torch.Tensor, lambd: float) -> torch.Tensor:
    """
    Applies the hard shrink function to an input tensor using bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output = torch.where(input_bf16.abs() > lambd, input_bf16, torch.zeros_like(input_bf16))
    return output.to(torch.float32)

function_signature = {
    "name": "torch_hardshrink_bfloat16_function",
    "inputs": [
        ((4, 4), torch.float32),
        (torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
