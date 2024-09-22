
import torch

def permute_and_add_function(input_tensor: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
    """
    Permute the input tensor and add a series of weights.
    """
    permuted = input_tensor.permute(1, 0, 2)
    for weight in weights:
        permuted = permuted + weight
    return permuted.permute(1, 0, 2)

function_signature = {
    "name": "permute_and_add_function",
    "inputs": [
        ((1, 2, 3), torch.float32),
        [((2, 3), torch.float32), ((2, 3), torch.float32)] 
    ],
    "outputs": [
        ((1, 2, 3), torch.float32),
    ]
}
