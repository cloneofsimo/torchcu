
import torch
import torch.nn.functional as F

def svd_multi_margin_loss_fp16(input_tensor: torch.Tensor, labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Calculates the multi-margin loss for a given input tensor using SVD and fp16 precision.

    Args:
        input_tensor: A 2D tensor of shape (batch_size, num_classes).
        labels: A 1D tensor of shape (batch_size) containing the true class labels.
        margin: The margin value for the multi-margin loss.

    Returns:
        A 1D tensor of shape (batch_size) containing the loss values for each sample.
    """
    input_tensor = input_tensor.to(torch.float16)
    labels = labels.to(torch.int8)
    
    # Calculate SVD
    U, S, V = torch.linalg.svd(input_tensor)

    # Extract the diagonal of S and multiply with V
    S = torch.diag(S)
    scores = torch.mm(S, V)
    
    # Calculate the multi-margin loss
    loss = F.multi_margin_loss(scores, labels, margin=margin, reduction='none')
    return loss.to(torch.float32)

function_signature = {
    "name": "svd_multi_margin_loss_fp16",
    "inputs": [
        ((10, 5), torch.float32),
        ((10,), torch.int8),
    ],
    "outputs": [
        ((10,), torch.float32),
    ]
}
