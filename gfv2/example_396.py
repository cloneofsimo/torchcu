
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(MyModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.instance_norm = nn.InstanceNorm2d(out_channels)

    def forward(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.instance_norm(x)
        x = torch.sqrt(x)
        x = torch.median(x, dim=1, keepdim=True).values
        return x

def my_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a sequence of operations including convolution, instance normalization, 
    square root, median calculation, and returns the result.
    """
    module = MyModule(in_channels=1, out_channels=8, kernel_size=3)
    output = module(input_tensor)
    return output

function_signature = {
    "name": "my_function",
    "inputs": [
        ((2, 1, 16, 16), torch.float32)
    ],
    "outputs": [
        ((2, 1, 16, 16), torch.float32),
    ]
}
