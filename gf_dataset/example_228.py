
import torch
import torch.nn as nn
from cutlass import *

class GeGLU(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return self.dropout(x * torch.sigmoid(gate))

def torch_geglu_layernorm_conv_function(input_tensor: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor, bias: torch.Tensor, conv_weight: torch.Tensor) -> torch.Tensor:
    """
    Perform GeGLU, Layer Normalization, and Lightweight Convolution
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight1_bf16 = weight1.to(torch.bfloat16)
    weight2_bf16 = weight2.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)
    conv_weight_bf16 = conv_weight.to(torch.bfloat16)

    # GeGLU
    x = GeGLU(input_tensor.shape[-1])(input_bf16)

    # Layer Normalization
    ln = nn.LayerNorm(x.shape[-1])
    x = ln(x).to(torch.float32)

    # Lightweight Convolution
    conv = nn.Conv1d(x.shape[-1], x.shape[-1], kernel_size=3, padding=1, groups=x.shape[-1])
    x = conv(x.permute(0, 2, 1)).permute(0, 2, 1).to(torch.bfloat16)

    # Regularization
    x = x + torch.randn_like(x) * 0.1

    return x.to(torch.float32)


function_signature = {
    "name": "torch_geglu_layernorm_conv_function",
    "inputs": [
        ((4, 4, 1024), torch.float32),
        ((2048, 1024), torch.float32),
        ((2048, 1024), torch.float32),
        ((1024,), torch.float32),
        ((1024, 1024, 3), torch.float32)
    ],
    "outputs": [
        ((4, 4, 1024), torch.float32),
    ]
}
