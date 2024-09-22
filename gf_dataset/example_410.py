
import torch
from torch.nn.functional import scharr
from torch.utils.checkpoint import checkpoint

def contrastive_loss_with_scharr_bf16(input1: torch.Tensor, input2: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Calculates a contrastive loss between two input tensors, using Scharr gradients as features.
    The Scharr gradient calculation and loss computation are performed in bfloat16 precision for efficiency.
    
    Args:
        input1: The first input tensor (N, C, H, W).
        input2: The second input tensor (N, C, H, W).
        temperature: The temperature parameter for the contrastive loss.
    
    Returns:
        A scalar tensor representing the contrastive loss.
    """

    # Calculate Scharr gradients in bfloat16
    input1_grad = scharr(input1.to(torch.bfloat16), 3).to(torch.bfloat16)
    input2_grad = scharr(input2.to(torch.bfloat16), 3).to(torch.bfloat16)

    # Calculate the cosine similarity between the gradients
    similarity = torch.cosine_similarity(input1_grad.flatten(1), input2_grad.flatten(1), dim=1)

    # Apply temperature scaling
    similarity = similarity / temperature

    # Apply the contrastive loss
    loss = -torch.logsumexp(similarity, dim=0)

    return loss.to(torch.float32)

function_signature = {
    "name": "contrastive_loss_with_scharr_bf16",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),
        ((1, 3, 224, 224), torch.float32),
        (torch.float32,),
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
