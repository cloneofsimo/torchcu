
import torch

def torch_addmv_function(input_tensor: torch.Tensor, weight: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Performs a matrix-vector product and adds it to the input tensor.
    """
    return torch.addmv(input_tensor, weight, vec)

function_signature = {
    "name": "torch_addmv_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
