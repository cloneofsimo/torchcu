
import torch

def torch_permute_and_add_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Permute the input tensor, add a weight tensor, and apply a ReLU activation.
    """
    input_tensor = input_tensor.permute(0, 2, 1)
    output = input_tensor + weight
    return torch.relu(output)

function_signature = {
    "name": "torch_permute_and_add_function",
    "inputs": [
        ((2, 3, 4), torch.float32),
        ((2, 4, 3), torch.float32)
    ],
    "outputs": [
        ((2, 4, 3), torch.float32),
    ]
}
