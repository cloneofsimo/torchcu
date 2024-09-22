
import torch
import torch.nn.functional as F

def embedding_loss_function(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute a hinge embedding loss.

    Args:
        input_tensor: Input tensor of shape (batch_size, embedding_dim).
        target_tensor: Target tensor of shape (batch_size,).

    Returns:
        A scalar tensor representing the hinge embedding loss.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    target_bf16 = target_tensor.to(torch.bfloat16)

    distances = F.pairwise_distance(input_bf16, input_bf16, p=2)
    positives = distances[target_bf16 == 1]
    negatives = distances[target_bf16 == 0]

    loss = torch.clamp(1 - negatives + positives, min=0).mean()
    return loss.to(torch.float32)

function_signature = {
    "name": "embedding_loss_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4,), torch.int64)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
