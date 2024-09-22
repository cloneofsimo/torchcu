
import torch
import torch.nn.functional as F

def adaptive_log_softmax_function(input_tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Applies adaptive log softmax to the input tensor.
    """
    return F.adaptive_log_softmax(input_tensor, dim=dim)

function_signature = {
    "name": "adaptive_log_softmax_function",
    "inputs": [
        ((10, 10), torch.float32),
        (1, torch.int32)
    ],
    "outputs": [
        ((10, 10), torch.float32)
    ]
}
