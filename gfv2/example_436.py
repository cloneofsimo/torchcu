
import torch
import torch.nn.functional as F
from typing import List, Tuple

class LightweightConv2d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(LightweightConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def my_complex_function(input_tensor: torch.Tensor, weights: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    This function performs a series of operations on the input tensor,
    including lightweight convolution, identity transformation, and bfloat16
    computation. It returns both the final output tensor and a list of intermediate
    tensors.

    Args:
        input_tensor: The input tensor, must have at least one dimension.
        weights: A list of tensors representing the weights for the convolutions.
        
    Returns:
        A tuple containing:
            - The final output tensor.
            - A list of intermediate tensors.
    """
    
    # Lightweight convolution
    conv_out = LightweightConv2d(in_channels=input_tensor.shape[1], out_channels=16)(input_tensor.to(torch.bfloat16)).to(torch.float32)
    
    # Identity transformation
    identity_out = conv_out
    
    # Bfloat16 computation
    bf16_out = F.relu(conv_out.to(torch.bfloat16)).to(torch.float32)
    
    # Store intermediate tensors
    intermediate_tensors = [conv_out, identity_out, bf16_out]

    # Final computation
    final_out = F.avg_pool2d(bf16_out, kernel_size=2, stride=2)

    return final_out, intermediate_tensors


function_signature = {
    "name": "my_complex_function",
    "inputs": [
        ((1, 3, 32, 32), torch.float32),
        [((16, 3, 3, 3), torch.float32), ((16, 16), torch.float32)] 
    ],
    "outputs": [
        ((1, 16, 16, 16), torch.float32),
        [((1, 16, 32, 32), torch.float32), ((1, 16, 32, 32), torch.float32), ((1, 16, 32, 32), torch.float32)],
    ]
}
