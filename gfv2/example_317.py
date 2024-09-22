
import torch

def fused_sigmoid_focal_loss_function(input_tensor: torch.Tensor, target_tensor: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Computes the sigmoid focal loss with fused operations for efficiency.

    Args:
        input_tensor: Tensor of shape (N, C, H, W) or (N, C) representing the model's predictions.
        target_tensor: Tensor of shape (N, C, H, W) or (N, C) representing the ground truth labels.
        alpha: Weighting factor for the positive class.
        gamma: Focusing parameter.

    Returns:
        Tensor of shape (N, C, H, W) or (N, C) representing the focal loss.
    """
    input_tensor = input_tensor.to(torch.float16)
    target_tensor = target_tensor.to(torch.float16)
    
    # Sigmoid activation
    p = torch.sigmoid(input_tensor)

    # Focal loss calculation
    loss = -alpha * (1 - p) ** gamma * target_tensor * torch.log(p) - (1 - alpha) * p ** gamma * (1 - target_tensor) * torch.log(1 - p)

    return loss.to(torch.float32)


function_signature = {
    "name": "fused_sigmoid_focal_loss_function",
    "inputs": [
        ((1, 1), torch.float32),
        ((1, 1), torch.float32),
        ((), torch.float32),
        ((), torch.float32)
    ],
    "outputs": [
        ((1, 1), torch.float32),
    ]
}

