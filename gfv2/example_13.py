
import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool3d

def center_loss_with_adaptive_pooling(features: torch.Tensor, labels: torch.Tensor, num_classes: int, 
                                       alpha: float = 0.5, size: int = 1) -> torch.Tensor:
    """
    Calculates the center loss with adaptive average pooling for a given batch of features and labels.

    Args:
        features: Batch of features, shape (batch_size, feature_dim)
        labels: Batch of labels, shape (batch_size,)
        num_classes: Number of classes
        alpha: Weighting factor for center loss
        size: Size for adaptive average pooling

    Returns:
        Center loss value
    """
    # Adaptive average pooling
    features = adaptive_avg_pool3d(features.reshape(-1, 1, features.shape[1], 1, 1), output_size=(size, size, size))
    features = features.squeeze()

    # Calculate class centers
    centers = nn.Parameter(torch.randn(num_classes, features.shape[1]))
    centers = centers.to(features.device)
    centers_batch = centers[labels]

    # Calculate center loss
    loss = torch.mean(torch.sum(torch.pow(features - centers_batch, 2), dim=1))
    loss = alpha * loss

    return loss

function_signature = {
    "name": "center_loss_with_adaptive_pooling",
    "inputs": [
        ((16, 128), torch.float32),
        ((16,), torch.int64),
        (10, torch.int64)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
