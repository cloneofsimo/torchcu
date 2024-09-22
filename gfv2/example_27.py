
import torch
import torch.nn.functional as F

def triplet_loss_with_attention(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, 
                                 attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Computes the triplet loss with attention weights.

    Args:
        anchor: (batch_size, embedding_dim) Anchor embeddings.
        positive: (batch_size, embedding_dim) Positive embeddings.
        negative: (batch_size, embedding_dim) Negative embeddings.
        attention_weights: (batch_size, 1) Attention weights for each sample.

    Returns:
        loss: (1,) The triplet loss.
    """

    # Apply attention weights to embeddings
    anchor = anchor * attention_weights
    positive = positive * attention_weights
    negative = negative * attention_weights

    # Calculate distances
    distance_ap = torch.sum((anchor - positive)**2, dim=1, keepdim=True)
    distance_an = torch.sum((anchor - negative)**2, dim=1, keepdim=True)

    # Apply margin ranking loss
    loss = F.margin_ranking_loss(distance_an, distance_ap, margin=1.0)
    return loss

function_signature = {
    "name": "triplet_loss_with_attention",
    "inputs": [
        ((16, 128), torch.float32),
        ((16, 128), torch.float32),
        ((16, 128), torch.float32),
        ((16, 1), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
