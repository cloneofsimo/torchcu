
import torch

def conv2d_with_gt_backward_return_one(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    gt: torch.Tensor
) -> torch.Tensor:
    """
    Performs a 2D convolution with a specified weight and bias,
    calculates the loss using Mean Squared Error (MSE) with the ground truth,
    and returns only the loss gradient.

    Args:
        input_tensor: Input tensor of shape (batch_size, in_channels, height, width).
        weight: Convolution kernel of shape (out_channels, in_channels, kernel_height, kernel_width).
        bias: Bias tensor of shape (out_channels).
        gt: Ground truth tensor of shape (batch_size, out_channels, height, width).

    Returns:
        A tensor containing the gradient of the loss with respect to the input tensor.
    """

    output = torch.nn.functional.conv2d(input_tensor, weight, bias, padding=1)  # Assuming padding = 1 for simplicity
    loss = torch.nn.functional.mse_loss(output, gt)
    loss.backward(retain_graph=True)  # Retain the graph for backpropagation
    return input_tensor.grad

function_signature = {
    "name": "conv2d_with_gt_backward_return_one",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),  # Input tensor
        ((16, 3, 3, 3), torch.float32),  # Weight tensor
        ((16,), torch.float32),  # Bias tensor
        ((1, 16, 224, 224), torch.float32),  # Ground truth tensor
    ],
    "outputs": [
        ((1, 3, 224, 224), torch.float32),  # Gradient of the loss
    ]
}
