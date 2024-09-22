
import torch
import torch.nn as nn
import torch.nn.functional as F

def scatter_add_conv2d_backprop(input_tensor: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor, output_shape: list) -> torch.Tensor:
    """
    Performs a scatter-add operation followed by a separable 2D convolution and backpropagation.

    Args:
        input_tensor (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).
        weights (torch.Tensor): Weights for the separable convolution.
        indices (torch.Tensor): Indices for the scatter-add operation.
        output_shape (list): Desired shape of the output tensor.

    Returns:
        torch.Tensor: Gradient of the input tensor.
    """
    # Scatter-add
    output = torch.zeros(output_shape, device=input_tensor.device)
    output.scatter_add_(0, indices, input_tensor)

    # Separable convolution
    output = F.conv2d(output, weights[0], padding=1, groups=output_shape[1])
    output = F.conv2d(output, weights[1])

    # Backpropagation through the convolution
    grad_input = F.conv_transpose2d(output, weights[1], bias=None, padding=1)
    grad_input = F.conv_transpose2d(grad_input, weights[0], bias=None, padding=1, groups=output_shape[1])

    # Backpropagation through the scatter-add
    grad_input = torch.gather(grad_input, 0, indices)

    return grad_input

function_signature = {
    "name": "scatter_add_conv2d_backprop",
    "inputs": [
        ((1, 16, 32, 32), torch.float32),
        ((2, 16, 3, 3), torch.float32),
        ((1, 16, 32, 32), torch.int64),
        ((1, 16, 32, 32), torch.int64)
    ],
    "outputs": [
        ((1, 16, 32, 32), torch.float32)
    ]
}
