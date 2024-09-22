
import torch
import torch.nn.functional as F

class MyModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(MyModule, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x, y):
        x = self.conv2d(x)
        z = F.fft_conv2d(x, y, padding="same")
        z = z.abs()
        z = z.to(torch.bfloat16)
        return z


def torch_module_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs convolution with FFT, elementwise summation, absolute value, and bfloat16 conversion.
    """
    model = MyModule(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    output = model(input_tensor, weight)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_module_function",
    "inputs": [
        ((3, 32, 32), torch.float32),
        ((16, 3, 3), torch.float32)
    ],
    "outputs": [
        ((16, 32, 32), torch.float32),
    ]
}
