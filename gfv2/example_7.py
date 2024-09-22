
import torch
import torch.nn.functional as F

def depthwise_conv2d_relu_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Perform a depthwise convolution with ReLU activation.
    """
    output = F.conv2d(input_tensor, weight, bias, groups=input_tensor.shape[1], padding=1)
    return F.relu(output)

function_signature = {
    "name": "depthwise_conv2d_relu_function",
    "inputs": [
        ((1, 16, 10, 10), torch.float32),
        ((16, 1, 3, 3), torch.float32),
        ((16,), torch.float32),
    ],
    "outputs": [
        ((1, 16, 10, 10), torch.float32),
    ]
}
