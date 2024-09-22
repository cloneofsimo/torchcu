
import torch
import torch.nn.functional as F

def contrastive_loss_example(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """
    Calculates the contrastive loss for a single anchor-positive-negative triplet.
    This loss encourages the anchor and positive to be close and the anchor and negative to be far apart.
    """
    distance_ap = F.pairwise_distance(anchor, positive)
    distance_an = F.pairwise_distance(anchor, negative)
    loss = torch.maximum(torch.tensor(0.0), distance_ap - distance_an + 1.0)  # Margin of 1.0
    return loss

function_signature = {
    "name": "contrastive_loss_example",
    "inputs": [
        ((1,), torch.float32),
        ((1,), torch.float32),
        ((1,), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
