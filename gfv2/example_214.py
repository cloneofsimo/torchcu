
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Module

class ChannelAttention(Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels // reduction_ratio),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channels // reduction_ratio, in_channels)
        )

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = self.mlp(avg_out.view(avg_out.size(0), -1))
        max_out = self.mlp(max_out.view(max_out.size(0), -1))
        out = torch.sigmoid(avg_out + max_out).view(x.size(0), self.in_channels, 1, 1)
        return x * out

class LocalAttention(Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super(LocalAttention, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        query = self.conv(x)
        key = x
        value = x
        attn = F.softmax(torch.matmul(query, key.transpose(1, 2)), dim=-1)
        out = torch.matmul(attn, value.transpose(1, 2)).transpose(1, 2)
        return out

class PixelShuffle(Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return torch.nn.functional.pixel_shuffle(x, self.upscale_factor)

def fancy_transform(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    A function demonstrating various operations, including channel attention, local attention, 
    pixel shuffle, bfloat16, int8, inplace operations, and a custom layer.
    """
    # Channel Attention (bfloat16)
    input_bf16 = input_tensor.to(torch.bfloat16)
    channel_attn = ChannelAttention(in_channels=input_bf16.shape[1])
    channel_attn_out = channel_attn(input_bf16).to(torch.float32)

    # Local Attention (int8)
    local_attn = LocalAttention(in_channels=channel_attn_out.shape[1])
    local_attn_out = local_attn(channel_attn_out.to(torch.int8)).to(torch.float32)

    # Pixel Shuffle (inplace)
    local_attn_out.data = F.pixel_shuffle(local_attn_out.data, upscale_factor=2)

    # Custom layer
    class MyCustomLayer(Module):
        def __init__(self, in_channels, out_channels):
            super(MyCustomLayer, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x)

    custom_layer = MyCustomLayer(in_channels=local_attn_out.shape[1], out_channels=64)
    custom_layer_out = custom_layer(local_attn_out)

    return custom_layer_out

function_signature = {
    "name": "fancy_transform",
    "inputs": [
        ((1, 3, 8, 8), torch.float32)
    ],
    "outputs": [
        ((1, 64, 16, 16), torch.float32)
    ]
}
