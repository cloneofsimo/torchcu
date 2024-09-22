
import torch
import torch.nn.functional as F

def triplet_loss_zero_crossing_rate(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """
    Calculates the triplet loss and zero-crossing rate for a batch of triplets.

    Args:
        anchor: Anchor tensor of shape (batch_size, embedding_dim).
        positive: Positive tensor of shape (batch_size, embedding_dim).
        negative: Negative tensor of shape (batch_size, embedding_dim).

    Returns:
        loss: Triplet loss value.
        zero_crossing_rate: Zero-crossing rate of the differences between anchor and positive embeddings.
    """
    # Calculate the distances between embeddings
    ap_dist = torch.norm(anchor - positive, dim=1)
    an_dist = torch.norm(anchor - negative, dim=1)

    # Calculate the triplet loss
    loss = F.relu(ap_dist - an_dist + 1.0)

    # Calculate the zero-crossing rate
    diff = anchor - positive
    zero_crossing_rate = (torch.abs(diff[:, 1:] - diff[:, :-1]) > 0).float().mean(dim=1)

    return loss.mean(), zero_crossing_rate.mean()

function_signature = {
    "name": "triplet_loss_zero_crossing_rate",
    "inputs": [
        ((16, 128), torch.float32),
        ((16, 128), torch.float32),
        ((16, 128), torch.float32)
    ],
    "outputs": [
        ((), torch.float32),
        ((), torch.float32)
    ]
}
