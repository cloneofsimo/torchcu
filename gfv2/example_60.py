
import torch
import torch.nn as nn

def contrastive_pooling(input_tensor: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Performs contrastive pooling on input tensor with given labels. 
    
    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps).
        labels (torch.Tensor): Labels tensor of shape (batch_size).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, channels).
    """
    # Adaptive max pooling to reduce time dimension
    pooled_output = nn.AdaptiveMaxPool1d(1)(input_tensor)
    pooled_output = pooled_output.squeeze(dim=-1)
    
    # Supervised contrastive loss
    contrastive_loss = nn.SupervisedContrastiveLoss(temperature=0.1)
    loss = contrastive_loss(pooled_output, labels)
    
    # In-place gradient computation
    loss.backward()
    
    # Return only the gradient of the pooled output
    return pooled_output.grad

function_signature = {
    "name": "contrastive_pooling",
    "inputs": [
        ((16, 128, 100), torch.float32),
        ((16,), torch.int64),
    ],
    "outputs": [
        ((16, 128), torch.float32),
    ]
}
