
import torch
import torch.nn as nn

def torch_arcface_loss_function(embeddings: torch.Tensor, labels: torch.Tensor,
                                 weight: torch.Tensor, margin: float,
                                 scale: float, orthogonal_reg_weight: float) -> torch.Tensor:
    """
    Calculates ArcFace loss with optional orthogonal regularization.
    """
    # Orthogonal regularization
    if orthogonal_reg_weight > 0:
        weight_norm = torch.linalg.norm(weight, ord=2, dim=1)
        orthogonal_reg_loss = (weight_norm.pow(2) - 1).pow(2).sum() * orthogonal_reg_weight
    else:
        orthogonal_reg_loss = 0.0

    # ArcFace loss calculation
    cosine = torch.nn.functional.cosine_similarity(embeddings, weight)
    phi = cosine * margin
    theta = torch.acos(cosine)
    new_theta = torch.where(theta <= torch.pi / 4, torch.cos(theta + phi), torch.cos(theta - phi))
    one_hot = torch.zeros_like(cosine)
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    output = torch.where(one_hot == 1, scale * new_theta, cosine)
    arcface_loss = -(torch.nn.functional.softmin(output, dim=1) * output).sum(dim=1).mean()
    
    # Combined loss
    loss = arcface_loss + orthogonal_reg_loss
    return loss

function_signature = {
    "name": "torch_arcface_loss_function",
    "inputs": [
        ((32, 128), torch.float32),
        ((32,), torch.int64),
        ((128, 1000), torch.float32),
        (float, torch.float32),
        (float, torch.float32),
        (float, torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
