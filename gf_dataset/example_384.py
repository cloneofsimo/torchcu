
import torch

def torch_permute_and_add_function(input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Permute the input tensor, multiply with weights, and add a scalar value.
    """
    permuted_input = input_tensor.permute(1, 0, 2)
    output = torch.matmul(permuted_input, weights)
    return output + 1.0

function_signature = {
    "name": "torch_permute_and_add_function",
    "inputs": [
        ((2, 3, 4), torch.float32),
        ((4, 3), torch.float32)
    ],
    "outputs": [
        ((2, 3), torch.float32)
    ]
}
