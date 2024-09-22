
import torch

def hinge_embedding_loss_quantized(input_tensor: torch.Tensor, target_tensor: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Calculates the hinge embedding loss for quantized inputs, returning the loss for each sample.
    
    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, embedding_dim).
        target_tensor (torch.Tensor): Target tensor of shape (batch_size, 1).
        margin (float, optional): Margin for the hinge loss. Defaults to 1.0.

    Returns:
        torch.Tensor: Loss for each sample, shape (batch_size,).
    """
    
    # Round input tensor to the nearest integer
    input_tensor_quantized = torch.round(input_tensor)
    
    # Calculate the hinge loss
    loss = torch.relu(margin + torch.mul(input_tensor_quantized, target_tensor) - 1.0)
    
    return loss

function_signature = {
    "name": "hinge_embedding_loss_quantized",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 1), torch.float32),
        (1.0, torch.float32)
    ],
    "outputs": [
        ((4,), torch.float32),
    ]
}
