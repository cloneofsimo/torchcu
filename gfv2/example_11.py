
import torch
import torch.nn as nn
from torch.nn import functional as F

class TokenMixing(nn.Module):
    def __init__(self, dim, seq_len, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, 1, d), qkv)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).reshape(b, n, d)
        x = self.to_out(x)
        return x

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, dim, seq_len):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, dim))

    def forward(self, x):
        return x + self.pos_embedding

class StochasticDepth(nn.Module):
    def __init__(self, p, mode="row"):
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x):
        if self.mode == "row":
            mask = (torch.rand(x.shape[0], 1, 1, device=x.device) < (1 - self.p)).float()
        elif self.mode == "channel":
            mask = (torch.rand(1, x.shape[1], 1, device=x.device) < (1 - self.p)).float()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return (x * mask) / (1 - self.p)

def token_mixing_function(input_tensor: torch.Tensor, weight_qkv: torch.Tensor, weight_out: torch.Tensor, norm_weight: torch.Tensor, norm_bias: torch.Tensor) -> torch.Tensor:
    """
    Performs token mixing operation with stochastic depth.
    """
    input_tensor = input_tensor.to(torch.int8)
    weight_qkv = weight_qkv.to(torch.int8)
    weight_out = weight_out.to(torch.int8)
    norm_weight = norm_weight.to(torch.int8)
    norm_bias = norm_bias.to(torch.int8)

    # Normalize input
    input_tensor = F.layer_norm(input_tensor, (input_tensor.shape[-1],), weight=norm_weight, bias=norm_bias)

    # Project to qkv
    qkv = F.linear(input_tensor, weight_qkv)
    q, k, v = qkv.chunk(3, dim=-1)

    # Calculate attention
    attn = (q * (input_tensor.shape[-1] ** -0.5)) @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)

    # Apply stochastic depth
    attn = StochasticDepth(p=0.1, mode="row")(attn)

    # Apply attention
    x = (attn @ v).reshape(input_tensor.shape)

    # Project to output
    x = F.linear(x, weight_out)

    return x.to(torch.float32)

function_signature = {
    "name": "token_mixing_function",
    "inputs": [
        ((8, 16, 64), torch.int8),
        ((192, 64), torch.int8),
        ((64, 64), torch.int8),
        ((64,), torch.int8),
        ((64,), torch.int8),
    ],
    "outputs": [
        ((8, 16, 64), torch.float32),
    ]
}
