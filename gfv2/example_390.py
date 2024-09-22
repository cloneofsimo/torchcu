
import torch

def triplet_loss_function(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Calculates the triplet margin loss.

    Args:
        anchor: Anchor tensor of shape (batch_size, embedding_dim).
        positive: Positive tensor of shape (batch_size, embedding_dim).
        negative: Negative tensor of shape (batch_size, embedding_dim).
        margin: Margin value.

    Returns:
        Triplet margin loss.
    """
    # Calculate distances
    anchor_positive_distance = torch.norm(anchor - positive, dim=1)
    anchor_negative_distance = torch.norm(anchor - negative, dim=1)

    # Apply triplet margin loss
    loss = torch.nn.functional.relu(anchor_positive_distance - anchor_negative_distance + margin)
    return torch.mean(loss)


function_signature = {
    "name": "triplet_loss_function",
    "inputs": [
        ((1, 8), torch.float32),
        ((1, 8), torch.float32),
        ((1, 8), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
