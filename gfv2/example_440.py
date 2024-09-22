
import torch
from torch import nn
import torch.nn.functional as F

def wasserstein_loss_bf16(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Wasserstein loss between two tensors using bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    target_bf16 = target_tensor.to(torch.bfloat16)
    loss = F.l1_loss(input_bf16, target_bf16)
    return loss.to(torch.float32)

function_signature = {
    "name": "wasserstein_loss_bf16",
    "inputs": [
        ((1,), torch.float32),  # Ensure at least one element in the input tensor
        ((1,), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
