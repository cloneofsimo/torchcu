
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MyModule, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_max_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)  # Flatten
        return x

def torch_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies a 3D convolution, activation, adaptive max pooling, and flattening.
    """
    model = MyModule(in_channels=3, out_channels=64)
    output = model(input_tensor)
    return output

function_signature = {
    "name": "torch_function",
    "inputs": [
        ((1, 3, 16, 16, 16), torch.float32)
    ],
    "outputs": [
        ((1, 4096), torch.float32),
    ]
}
