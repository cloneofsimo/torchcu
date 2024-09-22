
import torch
import torch.nn.functional as F

def simclr_loss_fp32(
    z1: torch.Tensor, 
    z2: torch.Tensor,
    temperature: float = 0.1,
    symmetric: bool = True
) -> torch.Tensor:
    """
    Compute the SimCLR loss for a batch of positive pairs (z1, z2).

    Args:
        z1: First set of representations, shape (batch_size, embedding_dim).
        z2: Second set of representations, shape (batch_size, embedding_dim).
        temperature: Temperature parameter for the contrastive loss.
        symmetric: Whether to compute the loss symmetrically.

    Returns:
        The contrastive loss, shape (batch_size,).
    """

    z1_norm = F.normalize(z1, p=2, dim=1)
    z2_norm = F.normalize(z2, p=2, dim=1)

    sim_matrix = torch.matmul(z1_norm, z2_norm.T)
    
    mask = torch.eye(z1.size(0), dtype=torch.bool)
    if not symmetric:
        mask = mask.triu(diagonal=1)

    sim_matrix = sim_matrix.masked_fill(mask, -torch.inf)

    positives = sim_matrix.diagonal()
    loss = -torch.log(torch.exp(positives / temperature) / torch.sum(torch.exp(sim_matrix / temperature), dim=1))

    return loss

function_signature = {
    "name": "simclr_loss_fp32",
    "inputs": [
        ((128, 128), torch.float32),
        ((128, 128), torch.float32),
        ((), torch.float32),
        ((), torch.bool),
    ],
    "outputs": [
        ((128,), torch.float32),
    ]
}

