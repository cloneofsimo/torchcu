
import torch
import torch.nn.functional as F

def torch_soft_margin_loss(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate the soft margin loss between input and target tensors.
    """
    return F.soft_margin_loss(input_tensor, target_tensor, reduction='none')

function_signature = {
    "name": "torch_soft_margin_loss",
    "inputs": [
        ((10,), torch.float32),
        ((10,), torch.float32)
    ],
    "outputs": [
        ((10,), torch.float32),
    ]
}
