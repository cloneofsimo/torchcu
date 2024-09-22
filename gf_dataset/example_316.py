
import torch
import torch.nn as nn
import torch.nn.functional as F

class StochasticDepthConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, p=0.2, bias=False):
        super(StochasticDepthConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, groups=groups, bias=bias)
        self.p = p

    def forward(self, x):
        if self.training:
            # Apply stochastic depth with probability p
            mask = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < (1 - self.p)).float()
            x = x * mask
        return self.conv(x)

def torch_stochastic_depth_grouped_conv_function(input_tensor: torch.Tensor, weight: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Performs a grouped convolution with stochastic depth, applying ReLU and exponential activation.
    """
    # Convert to bfloat16 for efficient computation
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)

    # Perform grouped convolution
    output = F.conv2d(input_bf16, weight_bf16, groups=groups)

    # Stochastic depth
    output = StochasticDepthConv(output.shape[1], output.shape[1], kernel_size=1, p=0.2)(output)

    # Apply ReLU and exp activation
    output = F.relu(output)
    output = torch.exp(output)

    # Return result in float32
    return output.to(torch.float32)

function_signature = {
    "name": "torch_stochastic_depth_grouped_conv_function",
    "inputs": [
        ((1, 64, 56, 56), torch.float32),
        ((32, 64, 3, 3), torch.float32),
        (8, torch.int32)
    ],
    "outputs": [
        ((1, 64, 56, 56), torch.float32),
    ]
}
