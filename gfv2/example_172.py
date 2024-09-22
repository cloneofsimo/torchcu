
import torch
import torch.nn as nn
from torch.nn.functional import normalize

def simclr_loss_fp16(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Compute SimCLR loss for two sets of embeddings.

    Args:
        z1: Embeddings of the first set of samples, shape (batch_size, embedding_dim).
        z2: Embeddings of the second set of samples, shape (batch_size, embedding_dim).

    Returns:
        SimCLR loss, a scalar tensor.
    """
    z1 = z1.to(torch.float16)
    z2 = z2.to(torch.float16)
    
    # Normalize embeddings
    z1_norm = normalize(z1, dim=1)
    z2_norm = normalize(z2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.mm(z1_norm, z2_norm.t())
    
    # Mask out diagonal elements (similarity of a sample with itself)
    mask = torch.eye(z1.shape[0], dtype=torch.bool)
    similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)
    
    # Calculate the maximum similarity (positive pair)
    positive_similarity = similarity_matrix.max(dim=1).values
    
    # Compute the negative similarity (the highest similarity among all negatives)
    negative_similarity = similarity_matrix.max(dim=1).values
    
    # Compute the loss
    loss = -torch.log(torch.exp(positive_similarity) / (torch.exp(positive_similarity) + torch.exp(negative_similarity))).mean()

    return loss.to(torch.float32)

function_signature = {
    "name": "simclr_loss_fp16",
    "inputs": [
        ((128, 128), torch.float32),
        ((128, 128), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
