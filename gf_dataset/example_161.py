
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(DynamicConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x, kernel_size):
        self.weight = self.weight[:, :, :kernel_size, :kernel_size]
        return F.conv2d(x, self.weight, self.bias, padding=(kernel_size // 2))

def torch_dynamic_conv_int8_function(input_tensor: torch.Tensor, weight: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Perform dynamic convolution using int8 quantization.
    """
    dynamic_conv = DynamicConv(input_tensor.shape[1], weight.shape[0], kernel_size)
    dynamic_conv.weight = weight
    output = dynamic_conv(input_tensor, kernel_size)
    return output.to(torch.int8)

function_signature = {
    "name": "torch_dynamic_conv_int8_function",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),
        ((16, 3, 3, 3), torch.float32),
        (3, torch.int32)
    ],
    "outputs": [
        ((1, 16, 224, 224), torch.int8),
    ]
}
