
import torch
import torch.nn as nn

class DepthwiseConv2d_BF16(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, bias=False, layer_scaling=1.0):
        super(DepthwiseConv2d_BF16, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.layer_scaling = layer_scaling

        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias:
            self.bias = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x):
        x = x.to(torch.bfloat16)
        weight = self.weight.to(torch.bfloat16)
        
        out = torch.nn.functional.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=self.in_channels, bias=None)
        out = out * self.layer_scaling
        
        if self.bias:
            out += self.bias.to(torch.bfloat16)
        out = out.to(torch.float32)
        return out

def depthwise_conv2d_bf16_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, layer_scaling: float) -> torch.Tensor:
    """
    Performs a depthwise convolution with bfloat16 precision.
    """
    
    # Input and weight are in bfloat16 format
    input_tensor = input_tensor.to(torch.bfloat16)
    weight = weight.to(torch.bfloat16)
    
    # Apply layer scaling
    layer_scaling = torch.tensor(layer_scaling).to(torch.bfloat16)
    
    # Compute the convolution
    output = torch.nn.functional.conv2d(input_tensor, weight, groups=input_tensor.shape[1], bias=None)
    output = output * layer_scaling
    
    # Add bias if provided
    if bias is not None:
        bias = bias.to(torch.bfloat16)
        output += bias
    
    # Return output in float32 format
    output = output.to(torch.float32)
    return output

function_signature = {
    "name": "depthwise_conv2d_bf16_function",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),
        ((3, 1, 3, 3), torch.float32),
        ((3,), torch.float32),
        ((1,), torch.float32)
    ],
    "outputs": [
        ((1, 3, 224, 224), torch.float32)
    ]
}
