
import torch

def triplet_loss_function(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Computes the triplet loss, a loss function used in face recognition and other similarity learning tasks.
    
    Args:
        anchor: A tensor representing the anchor embedding.
        positive: A tensor representing the positive embedding.
        negative: A tensor representing the negative embedding.
        margin: The margin parameter for the triplet loss.
    
    Returns:
        A scalar tensor representing the triplet loss.
    """
    distance_ap = torch.norm(anchor - positive, p=2, dim=1)
    distance_an = torch.norm(anchor - negative, p=2, dim=1)

    loss = torch.relu(distance_ap - distance_an + margin)
    return torch.mean(loss)


function_signature = {
    "name": "triplet_loss_function",
    "inputs": [
        ((16, 128), torch.float32),
        ((16, 128), torch.float32),
        ((16, 128), torch.float32),
        ((), torch.float32)
    ],
    "outputs": [
        ((), torch.float32)
    ]
}
