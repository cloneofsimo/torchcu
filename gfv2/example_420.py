
import torch
import torch.nn.functional as F
from torch import nn

class GeGLU(nn.Module):
    def __init__(self, dim, dropout=0.):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return self.dropout(x * torch.sigmoid(gate))

class MaskedAttention(nn.Module):
    def __init__(self, dim, n_heads, causal=False):
        super().__init__()
        assert dim % n_heads == 0, 'Dimensions must be divisible by number of heads'
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.causal = causal

    def forward(self, x, mask=None):
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(t.shape[0], t.shape[1], self.n_heads, self.head_dim).transpose(1, 2), qkv)

        # scaled dot product
        sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask[:, None, None, :]
            sim.masked_fill_(~mask, -torch.finfo(sim.dtype).max)
        if self.causal:
            i = torch.arange(sim.shape[1])[:, None]
            j = torch.arange(sim.shape[2])
            mask = i >= j
            sim.masked_fill_(mask, -torch.finfo(sim.dtype).max)
        attn = F.softmax(sim, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.n_heads * self.head_dim)

        return out


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        var = torch.var(x, dim=-1, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return self.g * (x - mean) / (var + self.eps) ** 0.5 + self.b

def my_function(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Example function with GeGLU, MaskedAttention, LayerNorm, and inplace operations.
    """
    x = LayerNorm(x.shape[-1])(x)
    x = GeGLU(x.shape[-1])(x)
    x = MaskedAttention(x.shape[-1], 4, causal=True)(x, mask)
    x = x + x
    return x

function_signature = {
    "name": "my_function",
    "inputs": [
        ((10, 16, 512), torch.float32),
        ((10, 16, 16), torch.bool),
    ],
    "outputs": [
        ((10, 16, 512), torch.float32),
    ]
}
