
import torch
import torch.nn as nn
from torch.nn.functional import normalize

def simclr_loss_function(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Calculates the SimCLR loss for a batch of positive pairs.

    Args:
        z1: Tensor of shape (batch_size, embedding_dim), representing the first view of each pair.
        z2: Tensor of shape (batch_size, embedding_dim), representing the second view of each pair.

    Returns:
        loss: The SimCLR loss for the batch.
    """
    z1 = normalize(z1, dim=1)
    z2 = normalize(z2, dim=1)
    
    similarity_matrix = z1 @ z2.T
    
    positives = torch.diag(similarity_matrix)
    
    mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool)
    negatives = similarity_matrix[~mask]
    
    loss = -torch.mean(torch.log(torch.exp(positives) / (torch.exp(positives) + torch.sum(torch.exp(negatives)))))
    
    return loss

function_signature = {
    "name": "simclr_loss_function",
    "inputs": [
        ((16, 128), torch.float32),
        ((16, 128), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
