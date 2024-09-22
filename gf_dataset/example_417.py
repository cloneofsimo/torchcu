
import torch
from torch import nn
import torch.nn.functional as F

class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x

def torch_multi_scale_attention_bfloat16_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs multi-scale attention with bfloat16 precision and a logsigmoid activation.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)

    msa = MultiScaleAttention(in_channels=input_tensor.shape[-1])
    output_bf16 = msa(input_bf16)
    output_bf16 = output_bf16.to(torch.float32)

    return torch.logsigmoid(output_bf16)

function_signature = {
    "name": "torch_multi_scale_attention_bfloat16_function",
    "inputs": [
        ((16, 128, 512), torch.float32),
        ((512, 512), torch.float32)
    ],
    "outputs": [
        ((16, 128, 512), torch.float32),
    ]
}
