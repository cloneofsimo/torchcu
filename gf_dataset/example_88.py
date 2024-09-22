
import torch
import torch.nn.functional as F

def triplet_margin_loss_with_interpolation(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0, p: float = 2) -> torch.Tensor:
    """
    Calculates the Triplet Margin Loss with interpolation.

    Args:
        anchor (torch.Tensor): Anchor tensor.
        positive (torch.Tensor): Positive tensor.
        negative (torch.Tensor): Negative tensor.
        margin (float, optional): Margin for the loss function. Defaults to 1.0.
        p (float, optional): Exponent value for the norm. Defaults to 2.

    Returns:
        torch.Tensor: Triplet Margin Loss.
    """
    
    # Calculate distances
    dist_pos = torch.norm(anchor - positive, p=p, dim=1)
    dist_neg = torch.norm(anchor - negative, p=p, dim=1)

    # Interpolate positive and negative features
    interpolated = (positive + negative) / 2.0

    # Calculate distance from anchor to interpolated features
    dist_interp = torch.norm(anchor - interpolated, p=p, dim=1)

    # Calculate loss
    loss = torch.maximum(torch.tensor(0.0), dist_pos - dist_neg + margin) + torch.maximum(torch.tensor(0.0), dist_interp - dist_neg + margin)

    return loss

function_signature = {
    "name": "triplet_margin_loss_with_interpolation",
    "inputs": [
        ((10, 128), torch.float32),
        ((10, 128), torch.float32),
        ((10, 128), torch.float32),
    ],
    "outputs": [
        ((10,), torch.float32),
    ]
}
