
import torch

def my_function(input_tensor: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on input tensor, weights and indices
    """
    input_tensor = input_tensor.to(torch.float16)
    weights = weights.to(torch.float16)
    indices = indices.to(torch.int8)
    
    output = torch.scatter_add(torch.zeros_like(weights, dtype=torch.float16), 0, indices, input_tensor * weights)
    output = output.abs()
    output = output.var(dim=0)
    output = torch.einsum("ij,jk->ik", output, weights)
    return output.to(torch.float32)

function_signature = {
    "name": "my_function",
    "inputs": [
        ((10,), torch.float32),
        ((10, 10), torch.float32),
        ((10,), torch.int32)
    ],
    "outputs": [
        ((10, 10), torch.float32)
    ]
}
