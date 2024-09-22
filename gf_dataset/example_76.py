
import torch

def torch_scatter_bfloat16_function(input_tensor: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs scatter operation on a tensor using bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output = torch.zeros_like(input_tensor, dtype=torch.bfloat16)
    output.scatter_(dim, index, input_bf16)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_scatter_bfloat16_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4,), torch.int64),
        (0, torch.int64)  # Scalar for dimension
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
