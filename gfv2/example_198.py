
import torch

def grouped_conv_relu_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Perform a grouped convolution with ReLU activation.
    """
    output = torch.nn.functional.conv2d(input_tensor, weight, bias, groups=groups)
    return torch.relu(output)

function_signature = {
    "name": "grouped_conv_relu_function",
    "inputs": [
        ((1, 3, 28, 28), torch.float32),  # input tensor
        ((16, 3, 3, 3), torch.float32),  # weight tensor
        ((16,), torch.float32),  # bias tensor
        (16,)  # groups
    ],
    "outputs": [
        ((1, 16, 26, 26), torch.float32)  # output tensor
    ]
}

