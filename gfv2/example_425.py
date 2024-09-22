
import torch
import torch.nn.functional as F

def lightweight_conv_pool_bce(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a lightweight convolution, average pooling, and binary cross-entropy loss calculation.
    """
    # Convolution
    output = F.conv2d(input_tensor.to(torch.int8), weight.to(torch.int8), bias=bias.to(torch.int8))

    # Average Pooling
    output = F.avg_pool2d(output, kernel_size=2)

    # Binary Cross-Entropy Loss
    loss = F.binary_cross_entropy(output, input_tensor.to(torch.float32))

    return loss

function_signature = {
    "name": "lightweight_conv_pool_bce",
    "inputs": [
        ((1, 1, 4, 4), torch.int8),
        ((1, 1, 3, 3), torch.int8),
        ((1,), torch.int8)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
