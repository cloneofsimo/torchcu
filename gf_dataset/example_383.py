
import torch

def torch_matrix_operations(input_tensor: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Performs a series of matrix operations, including sqrt, rank calculation, and dot product, 
    returning both the result of the dot product and the rank of the input tensor.
    """
    input_tensor_sqrt = torch.sqrt(input_tensor)
    rank = torch.matrix_rank(input_tensor_sqrt)
    output = torch.dot(input_tensor_sqrt, weight)
    return output, int(rank.item())

function_signature = {
    "name": "torch_matrix_operations",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
        ((), torch.int32)
    ]
}
