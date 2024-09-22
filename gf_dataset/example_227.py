
import torch
from torch import nn
from torch.nn import functional as F

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scale = self.head_dim**-0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

def torch_multihead_attention_fp16(input_tensor: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor) -> torch.Tensor:
    """
    Performs multi-head attention with fp16 precision.
    
    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
        weight1 (torch.Tensor): Weight for the first linear layer.
        weight2 (torch.Tensor): Weight for the second linear layer.

    Returns:
        torch.Tensor: Output tensor after attention operation.
    """
    input_tensor = input_tensor.to(torch.float16)
    weight1 = weight1.to(torch.float16)
    weight2 = weight2.to(torch.float16)

    attn_output = AttentionBlock(embed_dim=input_tensor.shape[-1], num_heads=4)(input_tensor)

    output = F.linear(attn_output, weight1.t())
    output = F.relu(output)
    output = F.linear(output, weight2.t())
    return output.to(torch.float32)


function_signature = {
    "name": "torch_multihead_attention_fp16",
    "inputs": [
        ((16, 128, 768), torch.float32),
        ((768, 768), torch.float32),
        ((768, 768), torch.float32)
    ],
    "outputs": [
        ((16, 128, 768), torch.float32),
    ]
}
