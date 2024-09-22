
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

def supervised_contrastive_loss_int8(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Calculates the supervised contrastive loss with int8 precision.

    Args:
        anchor: The anchor tensor (int8).
        positive: The positive tensor (int8).
        negative: The negative tensor (int8).
        temperature: The temperature parameter for the softmax.

    Returns:
        The supervised contrastive loss.
    """
    with autocast():
        similarity_matrix = torch.nn.functional.cosine_similarity(anchor.float(), torch.cat([positive.float(), negative.float()]), dim=1)
        positive_similarity = similarity_matrix[:, 0]
        negative_similarity = similarity_matrix[:, 1:]

        loss = torch.nn.functional.cross_entropy(
            (positive_similarity / temperature).unsqueeze(dim=1), 
            torch.zeros_like(positive_similarity).unsqueeze(dim=1), 
            reduction='none'
        )

        return loss

function_signature = {
    "name": "supervised_contrastive_loss_int8",
    "inputs": [
        ((16, 128), torch.int8),
        ((16, 128), torch.int8),
        ((16, 128), torch.int8),
    ],
    "outputs": [
        ((16,), torch.float32),
    ]
}
