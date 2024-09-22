
import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm

def torch_sparse_conv_kl_div(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, 
                             target_tensor: torch.Tensor, sparsity_weight: float) -> torch.Tensor:
    """
    Sparse convolutional operation with KL divergence regularization on weights.
    
    Args:
        input_tensor: Input tensor with shape (batch_size, in_channels, height, width).
        weight: Weight tensor with shape (out_channels, in_channels, kernel_size, kernel_size).
        bias: Bias tensor with shape (out_channels).
        target_tensor: Target tensor with shape (batch_size, out_channels, height, width).
        sparsity_weight: Weight for sparsity regularization.

    Returns:
        A tuple containing:
            - output: Output tensor with shape (batch_size, out_channels, height, width).
            - kl_div_loss: KL divergence loss for weight sparsity.
    """
    # Apply weight normalization
    weight = weight_norm(weight, dim=1)
    
    # Convolution operation
    output = F.conv2d(input_tensor, weight, bias, padding=1)
    
    # Calculate KL divergence for sparsity
    log_weights = torch.log(torch.abs(weight) + 1e-6)
    kl_div_loss = sparsity_weight * torch.sum(log_weights - weight)
    
    # Calculate cross-entropy loss
    loss = F.cross_entropy(output, target_tensor)
    
    # Backward propagation
    loss.backward()
    
    return output, kl_div_loss

function_signature = {
    "name": "torch_sparse_conv_kl_div",
    "inputs": [
        ((1, 3, 32, 32), torch.float32),
        ((16, 3, 3, 3), torch.float32),
        ((16,), torch.float32),
        ((1, 16, 32, 32), torch.int64),  # Target tensor with int64 dtype
        (None, torch.float32)  # Sparsity weight
    ],
    "outputs": [
        ((1, 16, 32, 32), torch.float32), 
        (None, torch.float32)
    ]
}
