
import torch

def torch_cumprod_bfloat16_function(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute the cumulative product along a given dimension using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output_bf16 = torch.cumprod(input_bf16, dim=dim)
    return output_bf16.to(torch.float32)

function_signature = {
    "name": "torch_cumprod_bfloat16_function",
    "inputs": [
        ((4, 4), torch.float32),
        (int, None),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
