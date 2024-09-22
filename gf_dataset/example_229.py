
import torch
import torch.nn.functional as F

def torch_nll_loss_function(input_tensor: torch.Tensor, target_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Calculate the negative log likelihood loss with optional weights.
    """
    return F.nll_loss(input_tensor, target_tensor, weight=weight)

function_signature = {
    "name": "torch_nll_loss_function",
    "inputs": [
        ((10, 3), torch.float32),
        ((10,), torch.long),
        ((3,), torch.float32)
    ],
    "outputs": [
        ((), torch.float32)
    ]
}
