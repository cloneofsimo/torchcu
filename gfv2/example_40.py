
import torch

def bfloat16_uniform_backward(input_tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Applies a uniform distribution with specified scale to input tensor using bfloat16,
    then computes the backward pass.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output = torch.rand_like(input_bf16, dtype=torch.bfloat16) * scale
    output.backward(torch.ones_like(output, dtype=torch.bfloat16))
    return input_tensor.grad.to(torch.float32)

function_signature = {
    "name": "bfloat16_uniform_backward",
    "inputs": [
        ((4, 4), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
