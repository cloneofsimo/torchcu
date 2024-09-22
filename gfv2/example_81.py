
import torch

def cosine_embedding_loss_example(input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor,
                                  margin: float = 0.0, reduction: str = "mean") -> torch.Tensor:
    """
    Calculates the cosine embedding loss between two input tensors.

    Args:
        input1: Tensor of shape (N, D) or (N, S, D), where N is the batch size, S is the sequence length, and D is the feature dimension.
        input2: Tensor of shape (N, D) or (N, S, D), matching the shape of input1.
        target: Tensor of shape (N) or (N, S), containing target values of -1 or 1.
        margin: Margin value for the loss calculation, defaults to 0.0.
        reduction: Specifies how the loss is reduced, one of 'mean', 'sum', or 'none'. Defaults to 'mean'.

    Returns:
        A tensor containing the loss value(s).
    """
    similarity = torch.einsum('nd,nd->n', input1, input2)
    # Calculate the cosine similarity
    cosine_sim = similarity / (torch.linalg.norm(input1, dim=1) * torch.linalg.norm(input2, dim=1))
    # Calculate the loss
    loss = torch.clamp(margin - target * cosine_sim, min=0.0)
    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

function_signature = {
    "name": "cosine_embedding_loss_example",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.int64),
        (0.0, torch.float32),
        ("mean", torch.str)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
