
import torch

def cross_entropy_with_bucketized_weights(input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, num_buckets: int) -> torch.Tensor:
    """
    Calculates cross entropy loss with bucketized weights.
    
    Args:
        input (torch.Tensor): The model's output, shape (batch_size, num_classes).
        target (torch.Tensor): The ground truth labels, shape (batch_size).
        weights (torch.Tensor): Weights for each class, shape (num_classes).
        num_buckets (int): Number of buckets for weight discretization.

    Returns:
        torch.Tensor: The cross entropy loss with bucketized weights.
    """
    # Bucketize weights
    bucket_boundaries = torch.linspace(weights.min(), weights.max(), num_buckets)
    bucketed_weights = torch.bucketize(weights, bucket_boundaries)
    
    # Scatter add bucketed weights to create weighted class distribution
    weighted_class_distribution = torch.zeros_like(input).scatter_add_(dim=1, index=target.unsqueeze(1), src=bucketed_weights)
    
    # Calculate cross entropy loss
    loss = torch.nn.functional.cross_entropy(input, target, weight=weighted_class_distribution)
    return loss

function_signature = {
    "name": "cross_entropy_with_bucketized_weights",
    "inputs": [
        ((128, 10), torch.float32),
        ((128,), torch.int64),
        ((10,), torch.float32),
        (10,)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
