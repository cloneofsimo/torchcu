
import torch
import torch.nn.functional as F

def arcface_loss(features: torch.Tensor, labels: torch.Tensor,  s: float = 64.0, m: float = 0.5,
                  easy_margin: bool = False) -> torch.Tensor:
    """
    Computes the ArcFace loss.

    Args:
        features: Embeddings of shape (batch_size, embedding_dim).
        labels: Target labels of shape (batch_size,).
        s: Scale factor.
        m: Margin.
        easy_margin: Whether to use the easy margin.

    Returns:
        The ArcFace loss.
    """
    cos_theta = F.cosine_similarity(features, features[labels])
    
    # Handle numerical instability for very small cos_theta values
    cos_theta = torch.clamp(cos_theta, min=-1.0 + 1e-7, max=1.0 - 1e-7)

    if easy_margin:
        theta = torch.acos(cos_theta)
        margin_cos_theta = torch.cos(theta + m)
        cond_v = cos_theta - margin_cos_theta
        cos_theta = torch.where(cond_v > 0, cos_theta, margin_cos_theta)
    else:
        cos_theta = cos_theta * (1.0 - m) + m

    one_hot = torch.zeros_like(cos_theta)
    one_hot.scatter_(1, labels.view(-1, 1), 1)

    output = -s * (one_hot * cos_theta)
    return output

function_signature = {
    "name": "arcface_loss",
    "inputs": [
        ((10, 128), torch.float32),
        ((10,), torch.int64),
    ],
    "outputs": [
        ((10,), torch.float32),
    ]
}

