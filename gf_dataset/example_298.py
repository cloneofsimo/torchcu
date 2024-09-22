
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

def torch_function(input_tensor: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations:
        1. Convolution with the given weights and bias
        2. Batch normalization
        3. ReLU activation
    """
    module = MyModule(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    module.conv.weight.data = weights
    module.conv.bias.data = bias
    module.bn.weight.data = torch.ones(16)
    module.bn.bias.data = torch.zeros(16)
    output = module(input_tensor)
    return output

function_signature = {
    "name": "torch_function",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),  # Input tensor
        ((16, 3, 3, 3), torch.float32),  # Weights
        ((16,), torch.float32)  # Bias
    ],
    "outputs": [
        ((1, 16, 224, 224), torch.float32)  # Output tensor
    ]
}

