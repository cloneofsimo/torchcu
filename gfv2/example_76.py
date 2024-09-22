
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MyModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = torch.true_divide(x, torch.mean(x, dim=[2, 3], keepdim=True))  # Normalize
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))  # Global average pooling
        return x

def my_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies a convolution, batch normalization, ReLU, normalization, and adaptive average pooling.
    """
    model = MyModule(in_channels=3, out_channels=16)
    output = model(input_tensor)
    return output

function_signature = {
    "name": "my_function",
    "inputs": [
        ((3, 32, 32), torch.float32)
    ],
    "outputs": [
        ((16, 1, 1), torch.float32)
    ]
}
