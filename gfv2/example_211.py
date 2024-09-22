
import torch
import torch.nn as nn
from torch.nn import functional as F

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x).view(b, n, h, self.dim_head).transpose(1, 2)
        k = self.to_k(x).view(b, n, h, self.dim_head).transpose(1, 2)
        v = self.to_v(x).view(b, n, h, self.dim_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask[:, None, None, :].expand_as(attn)
            attn.masked_fill_(~mask, -torch.finfo(attn.dtype).max)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, h * self.dim_head)
        out = self.to_out(out)
        return out

def my_function(input_tensor: torch.Tensor, weight: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    This function performs a linear attention operation followed by a cross-entropy loss calculation.
    """
    x = input_tensor.to(torch.float16)
    w = weight.to(torch.float16)
    l = label.long()
    out = LinearAttention(dim=x.shape[-1], heads=8, dim_head=64)(x)
    out = torch.einsum('bnh,bh->bn', out, w)
    loss = F.cross_entropy(out, l)
    return loss.to(torch.float32)


function_signature = {
    "name": "my_function",
    "inputs": [
        ((128, 1024), torch.float32),
        ((1024, 128), torch.float32),
        ((128,), torch.int64)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}

