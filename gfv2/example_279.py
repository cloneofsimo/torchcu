
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits

def contrastive_loss_with_sparsity(anchor_features: torch.Tensor, positive_features: torch.Tensor, negative_features: torch.Tensor, weight: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Computes a contrastive loss with sparsity regularization on weights.

    Args:
        anchor_features: Tensor of shape (batch_size, feature_dim) representing anchor features.
        positive_features: Tensor of shape (batch_size, feature_dim) representing positive features.
        negative_features: Tensor of shape (batch_size, num_negatives, feature_dim) representing negative features.
        weight: Tensor of shape (feature_dim,) representing the weights for each feature dimension.
        temperature: Temperature scaling factor for the contrastive loss.

    Returns:
        A scalar tensor representing the contrastive loss with sparsity regularization.
    """
    batch_size = anchor_features.size(0)
    num_negatives = negative_features.size(1)

    # Calculate similarity scores
    anchor_dot_positive = torch.sum(anchor_features * positive_features, dim=1, keepdim=True)
    anchor_dot_negatives = torch.matmul(anchor_features, negative_features.transpose(1, 2))

    # Apply temperature scaling
    similarity_matrix = torch.cat([anchor_dot_positive, anchor_dot_negatives], dim=1) / temperature

    # Calculate contrastive loss
    labels = torch.zeros(batch_size, dtype=torch.long).to(anchor_features.device)
    loss_contrastive = binary_cross_entropy_with_logits(similarity_matrix, labels, reduction='mean')

    # Calculate sparsity loss
    sparsity_loss = torch.sum(torch.abs(weight)) / weight.numel()
    
    # Combine losses
    total_loss = loss_contrastive + sparsity_loss
    return total_loss

function_signature = {
    "name": "contrastive_loss_with_sparsity",
    "inputs": [
        ((128, 1024), torch.float32),
        ((128, 1024), torch.float32),
        ((128, 10, 1024), torch.float32),
        ((1024,), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
