
import torch
import torch.nn as nn
from torch.nn.functional import adaptive_max_pool1d, upsample
from torch.nn.functional import normalize
from torch.nn.functional import layer_norm

def simclr_loss_function(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Computes the SimCLR loss.

    Args:
        z1: The first set of representations (N x D).
        z2: The second set of representations (N x D).

    Returns:
        The SimCLR loss.
    """
    z1 = normalize(z1, dim=1)
    z2 = normalize(z2, dim=1)

    similarity_matrix = torch.matmul(z1, z2.T)
    mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool)
    positive_similarity = similarity_matrix[mask].view(similarity_matrix.shape[0], -1)
    negative_similarity = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    loss = -torch.log(torch.exp(positive_similarity) / (torch.exp(positive_similarity) + torch.sum(torch.exp(negative_similarity), dim=1, keepdim=True)))
    return torch.mean(loss)

function_signature = {
    "name": "simclr_loss_function",
    "inputs": [
        ((128, 128), torch.float32),
        ((128, 128), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
