
import torch
import torch.nn.functional as F

def triplet_loss_int8_function(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Calculates the triplet loss with int8 precision.

    Args:
        anchor (torch.Tensor): The anchor tensor of shape (batch_size, embedding_dim).
        positive (torch.Tensor): The positive tensor of shape (batch_size, embedding_dim).
        negative (torch.Tensor): The negative tensor of shape (batch_size, embedding_dim).
        margin (float): The margin parameter for the triplet loss. Defaults to 1.0.

    Returns:
        torch.Tensor: The triplet loss value.
    """
    anchor_int8 = anchor.to(torch.int8)
    positive_int8 = positive.to(torch.int8)
    negative_int8 = negative.to(torch.int8)
    
    distance_ap = torch.sum((anchor_int8 - positive_int8)**2, dim=1)
    distance_an = torch.sum((anchor_int8 - negative_int8)**2, dim=1)
    
    loss = torch.max(distance_ap - distance_an + margin, torch.zeros_like(distance_ap))
    return loss.mean()

function_signature = {
    "name": "triplet_loss_int8_function",
    "inputs": [
        ((1, 10), torch.float32),
        ((1, 10), torch.float32),
        ((1, 10), torch.float32),
        (1.0, torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
