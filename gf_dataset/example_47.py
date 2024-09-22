
import torch
from torch import nn

class MyModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)

def torch_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a forward pass through a simple convolutional network, including:
    - Convolution with ReLU activation
    - Adaptive average pooling
    - Linear transformation
    - ReLU activation
    """
    model = MyModule(input_tensor.size(1), weight.size(0))
    model.conv1.weight = nn.Parameter(weight.to(torch.float16))
    model.conv1.bias = nn.Parameter(bias.to(torch.float16))
    model.conv2.weight = nn.Parameter(weight.to(torch.float16))
    model.conv2.bias = nn.Parameter(bias.to(torch.float16))
    output = model(input_tensor.to(torch.float16)).to(torch.float32)
    return output

function_signature = {
    "name": "torch_function",
    "inputs": [
        ((1, 3, 32, 32), torch.float32),
        ((16, 3, 3, 3), torch.float32),
        ((16,), torch.float32),
    ],
    "outputs": [
        ((1, 16), torch.float32),
    ]
}
