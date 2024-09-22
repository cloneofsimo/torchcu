
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.depthwise_conv2d = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
        self.layer_scaling = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = self.conv3d(x)
        x = F.relu(x)
        x = self.depthwise_conv2d(x)
        x = F.relu(x)
        x = x * self.layer_scaling
        return x

def torch_model_function(input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Applies a 3D convolution, depthwise convolution, ReLU activation, layer scaling, and NLL loss.
    """
    model = MyModel(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
    output = model(input_tensor)
    loss = F.nll_loss(output, target)
    return loss

function_signature = {
    "name": "torch_model_function",
    "inputs": [
        ((1, 1, 28, 28, 28), torch.float32),
        ((1,), torch.int64)
    ],
    "outputs": [
        ((), torch.float32),
    ]
}
