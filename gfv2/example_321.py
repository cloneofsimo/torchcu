
import torch
import torch.nn as nn
import torch.nn.functional as F

def regularized_einsum_pooling(input_tensor: torch.Tensor, weight: torch.Tensor, 
                                 pool_size: int, reg_lambda: float) -> torch.Tensor:
    """
    Performs a regularized einsum-based pooling operation.

    Args:
        input_tensor: Input tensor of shape (batch_size, channels, height, width).
        weight: Weight tensor of shape (channels, channels).
        pool_size: Size of the pooling window.
        reg_lambda: Regularization strength.

    Returns:
        Output tensor of shape (batch_size, channels, pooled_height, pooled_width).
    """

    # Apply regularization to weight
    reg_loss = reg_lambda * torch.sum(weight ** 2)

    # Perform einsum-based matrix multiplication for channel-wise attention
    output = torch.einsum('bchw,cc->bchw', input_tensor, weight)

    # Apply adaptive max pooling
    output = F.adaptive_max_pool2d(output, (pool_size, pool_size))

    return output, reg_loss

function_signature = {
    "name": "regularized_einsum_pooling",
    "inputs": [
        ((1, 3, 16, 16), torch.float32),
        ((3, 3), torch.float32),
        (4, ), torch.int32,
        (1.0, ), torch.float32
    ],
    "outputs": [
        ((1, 3, 4, 4), torch.float32),
        (1.0, ), torch.float32
    ]
}
