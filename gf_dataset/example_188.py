
import torch
import torch.nn.functional as F
from torch.nn.functional import elu
from torch.nn import Conv2d
import torch.nn as nn
from cutlass import *

class GlobalAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(GlobalAttentionModule, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Calculate global attention weights
        attention_weights = self.conv(x)
        attention_weights = self.softmax(attention_weights)

        # Apply attention weights to input
        attended_x = attention_weights * x

        return attended_x

def torch_global_attention_elu_morphological_opening_fp16(input_tensor: torch.Tensor, weight: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Applies global attention, elu activation, morphological opening, and converts to fp16.
    """
    # Global attention
    global_attention = GlobalAttentionModule(input_tensor.shape[1], input_tensor.shape[1])
    attended_input = global_attention(input_tensor.to(torch.float16))

    # ELU activation
    elu_output = elu(attended_input)

    # Morphological opening (using a square kernel)
    kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float16)
    morphological_opening = F.max_pool2d(elu_output, kernel_size=kernel.shape, stride=1, padding=kernel.shape[0] // 2)
    morphological_opening = F.min_pool2d(morphological_opening, kernel_size=kernel.shape, stride=1, padding=kernel.shape[0] // 2)

    # Convert to fp16
    return morphological_opening.to(torch.float16)

function_signature = {
    "name": "torch_global_attention_elu_morphological_opening_fp16",
    "inputs": [
        ((1, 128, 256, 256), torch.float32),
        ((128, 128, 3, 3), torch.float32),
        (3, torch.int32)
    ],
    "outputs": [
        ((1, 128, 256, 256), torch.float16),
    ]
}
