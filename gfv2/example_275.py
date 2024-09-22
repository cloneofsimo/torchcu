
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.scale = self.head_dim ** -0.5
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        return x

def attention_square(x: torch.Tensor) -> torch.Tensor:
    """
    Performs attention and squares the result.
    """
    x = x.to(torch.float32)
    x = Attention(embed_dim=x.shape[-1], num_heads=4)(x)
    x = x * x  # element-wise squaring
    return x.to(torch.int8)

function_signature = {
    "name": "attention_square",
    "inputs": [
        ((1, 10, 128), torch.float32)
    ],
    "outputs": [
        ((1, 10, 128), torch.int8),
    ]
}
