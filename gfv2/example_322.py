
import torch
import torch.nn as nn
import torch.nn.functional as F

def depthwise_separable_conv_with_clipping(input_tensor: torch.Tensor, depthwise_weight: torch.Tensor, pointwise_weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    Performs a depthwise separable convolution with gradient clipping.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (N, C_in, H, W).
        depthwise_weight (torch.Tensor): Depthwise convolution weight of shape (C_in, 1, k_h, k_w).
        pointwise_weight (torch.Tensor): Pointwise convolution weight of shape (C_out, C_in, 1, 1).
        bias (torch.Tensor, optional): Bias for the pointwise convolution. Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (N, C_out, H_out, W_out).
    """
    # Depthwise convolution
    output = F.conv2d(input_tensor, depthwise_weight, groups=input_tensor.shape[1], padding='same')
    
    # Pointwise convolution
    output = F.conv2d(output, pointwise_weight, bias=bias)
    
    # ReLU activation
    output = F.relu(output)
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(output, 1.0)
    
    return output

function_signature = {
    "name": "depthwise_separable_conv_with_clipping",
    "inputs": [
        ((1, 3, 32, 32), torch.float32),
        ((3, 1, 3, 3), torch.float32),
        ((16, 3, 1, 1), torch.float32),
        ((16,), torch.float32),
    ],
    "outputs": [
        ((1, 16, 32, 32), torch.float32),
    ]
}
