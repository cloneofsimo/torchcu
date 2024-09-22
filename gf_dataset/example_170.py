
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(MyModule, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

def torch_module_forward(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs the forward pass of a 3D convolutional module with max pooling.
    """
    module = MyModule(in_channels=3, out_channels=64, kernel_size=3, stride=1)
    module.to(torch.float16)
    with torch.cuda.amp.autocast():
        output = module(input_tensor.to(torch.float16))
    return output.to(torch.float32)

function_signature = {
    "name": "torch_module_forward",
    "inputs": [
        ((1, 3, 16, 16, 16), torch.float32)
    ],
    "outputs": [
        ((1, 64, 8, 8, 8), torch.float32)
    ]
}
