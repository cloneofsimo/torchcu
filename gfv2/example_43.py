
import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss_with_transposed_conv1d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    target_tensor: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Computes a contrastive loss with a transposed convolutional layer.

    Args:
        input_tensor: Input tensor of shape (batch_size, in_channels, in_length).
        weight: Weight tensor of shape (out_channels, in_channels, kernel_size).
        bias: Bias tensor of shape (out_channels).
        target_tensor: Target tensor of shape (batch_size, out_channels, out_length).
        margin: Margin for the contrastive loss.

    Returns:
        A scalar tensor representing the contrastive loss.
    """
    # Apply transposed convolution
    output = F.conv_transpose1d(input_tensor, weight, bias)
    
    # Calculate contrastive loss
    loss = F.margin_ranking_loss(
        output.view(-1),
        target_tensor.view(-1),
        torch.ones_like(target_tensor.view(-1)),
        margin=margin,
    )
    
    return loss

function_signature = {
    "name": "contrastive_loss_with_transposed_conv1d",
    "inputs": [
        ((16, 16, 32), torch.float32),
        ((32, 16, 5), torch.float32),
        ((32,), torch.float32),
        ((16, 32, 64), torch.float32),
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
