
import torch

def torch_baddbmm_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Perform a batched matrix multiplication, add bias, and apply ReLU activation.
    """
    return torch.nn.functional.relu(torch.baddbmm(input_tensor, weight, bias))

function_signature = {
    "name": "torch_baddbmm_function",
    "inputs": [
        ((2, 3, 4), torch.float32),
        ((3, 4), torch.float32),
        ((3,), torch.float32)
    ],
    "outputs": [
        ((2, 3, 4), torch.float32),
    ]
}
