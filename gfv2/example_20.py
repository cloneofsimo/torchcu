
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

def torch_center_loss_function(input_tensor: torch.Tensor, centers: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Calculates the center loss for a batch of features, with bucketing and weighted averaging.
    """
    # Bucketize features
    bucket_indices = torch.bucketize(input_tensor, torch.linspace(0, 1, 10))  # 10 buckets
    
    # Compute center loss for each bucket
    bucket_losses = []
    for i in range(10):
        indices = (bucket_indices == i)
        features_in_bucket = input_tensor[indices]
        center_in_bucket = centers[i]
        loss_in_bucket = torch.sum((features_in_bucket - center_in_bucket) ** 2)
        bucket_losses.append(loss_in_bucket)

    # Weighted average of bucket losses
    weighted_losses = [loss * weight for loss, weight in zip(bucket_losses, weights)]
    total_loss = torch.sum(torch.stack(weighted_losses))
    
    return total_loss

function_signature = {
    "name": "torch_center_loss_function",
    "inputs": [
        ((128, 512), torch.float32),  # Features
        ((10, 512), torch.float32),  # Centers
        ((10,), torch.float32),  # Weights
    ],
    "outputs": [
        ((), torch.float32),  # Total center loss
    ]
}
