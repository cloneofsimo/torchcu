
import torch
import torch.nn.functional as F

def torch_soft_margin_loss_function(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the soft margin loss between input and target tensors.
    """
    return F.soft_margin_loss(input_tensor, target_tensor)

function_signature = {
    "name": "torch_soft_margin_loss_function",
    "inputs": [
        ((10,), torch.float32),
        ((10,), torch.float32)
    ],
    "outputs": [
        ((), torch.float32),
    ]
}
