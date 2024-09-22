
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm2d, Upsample

def torch_triplet_loss_bn_upsample_bf16(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Computes the triplet loss with batch normalization, upsampling, and bfloat16 precision.

    Args:
        anchor: Anchor tensor of shape (B, C, H, W).
        positive: Positive tensor of shape (B, C, H, W).
        negative: Negative tensor of shape (B, C, H, W).
        margin: Margin for the triplet loss.

    Returns:
        Triplet loss value.
    """
    # Batch Normalization
    bn = BatchNorm2d(anchor.shape[1])
    anchor = bn(anchor.to(torch.bfloat16))
    positive = bn(positive.to(torch.bfloat16))
    negative = bn(negative.to(torch.bfloat16))

    # Upsampling
    upsample = Upsample(scale_factor=2)
    anchor = upsample(anchor)
    positive = upsample(positive)
    negative = upsample(negative)

    # Triplet Loss
    distance_ap = torch.sum((anchor - positive)**2, dim=[1, 2, 3])
    distance_an = torch.sum((anchor - negative)**2, dim=[1, 2, 3])
    loss = torch.relu(distance_ap - distance_an + margin)

    return loss.mean().to(torch.float32)

function_signature = {
    "name": "torch_triplet_loss_bn_upsample_bf16",
    "inputs": [
        ((16, 3, 64, 64), torch.float32),
        ((16, 3, 64, 64), torch.float32),
        ((16, 3, 64, 64), torch.float32),
        (1.0, torch.float32),
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
