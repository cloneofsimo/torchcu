
import torch

def diagflat_relu_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies ReLU to the diagonal of a tensor.
    """
    diag = torch.diagflat(input_tensor)
    return torch.relu(diag)

function_signature = {
    "name": "diagflat_relu_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
