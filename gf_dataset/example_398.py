
import torch
import torch.nn.functional as F

def torch_triplet_loss_bf16(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float) -> torch.Tensor:
    """
    Calculates the triplet loss with bfloat16 precision.
    """
    anchor_bf16 = anchor.to(torch.bfloat16)
    positive_bf16 = positive.to(torch.bfloat16)
    negative_bf16 = negative.to(torch.bfloat16)

    # Calculate distances
    dist_pos = torch.norm(anchor_bf16 - positive_bf16, dim=1, p=2).to(torch.float32)
    dist_neg = torch.norm(anchor_bf16 - negative_bf16, dim=1, p=2).to(torch.float32)

    # Triplet loss
    loss = F.relu(dist_pos - dist_neg + margin)

    return loss

function_signature = {
    "name": "torch_triplet_loss_bf16",
    "inputs": [
        ((10, 128), torch.float32),
        ((10, 128), torch.float32),
        ((10, 128), torch.float32),
        (1, torch.float32)
    ],
    "outputs": [
        ((10,), torch.float32)
    ]
}

