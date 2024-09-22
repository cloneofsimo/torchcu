
import torch

def loss_function(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates both L1 and MSE loss between two tensors.
    """
    l1_loss = torch.nn.functional.l1_loss(input_tensor, target_tensor)
    mse_loss = torch.nn.functional.mse_loss(input_tensor, target_tensor)
    return torch.stack([l1_loss, mse_loss])

function_signature = {
    "name": "loss_function",
    "inputs": [
        ((10,), torch.float32),
        ((10,), torch.float32)
    ],
    "outputs": [
        ((2,), torch.float32),
    ]
}
