
import torch
import torch.nn.functional as F
from torch.nn import Dropout
from torch.nn.modules.activation import Softmax
import torch.cuda.amp as amp

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.dropout = Dropout(dropout)
        self.qkv = torch.nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        return x

def torch_multihead_attention_fp16(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Multi-head attention with FP16 precision and cudnn backend.
    """
    with amp.autocast():
        attention = MultiHeadAttention(embed_dim=query.shape[-1], num_heads=8)
        output = attention(query, mask=mask)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_multihead_attention_fp16",
    "inputs": [
        ((10, 10, 1024), torch.float32),  # Query
        ((10, 10, 1024), torch.float32),  # Key
        ((10, 10, 1024), torch.float32),  # Value
        ((10, 10), torch.bool)  # Mask
    ],
    "outputs": [
        ((10, 10, 1024), torch.float32)  # Output
    ]
}
