
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeGLU(nn.Module):
    def __init__(self, dim, ff_mult=4):
        super().__init__()
        self.proj = nn.Linear(dim, dim * ff_mult)
        self.gate = nn.Linear(dim, dim * ff_mult)

    def forward(self, x):
        x = self.proj(x)
        gate = torch.sigmoid(self.gate(x))
        return x * gate

class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, ff_mult=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = GeGLU(dim, ff_mult)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + x
        return x

def transformer_encoder_int8_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a transformer encoder operation using int8 quantization and bfloat16 computation.

    Args:
        input_tensor: Input tensor with shape (batch_size, sequence_length, dim).
        weight: Weights for the linear layers with shape (dim, dim).
        bias: Biases for the linear layers with shape (dim,).

    Returns:
        Output tensor with shape (batch_size, sequence_length, dim).
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    # Simulate a transformer encoder
    x = input_bf16
    x = F.linear(x, weight_bf16, bias_bf16)
    x = F.relu(x)

    x = F.linear(x, weight_bf16, bias_bf16)
    x = F.relu(x)

    x = F.linear(x, weight_bf16, bias_bf16)
    x = F.relu(x)

    return x.to(torch.float32)

function_signature = {
    "name": "transformer_encoder_int8_function",
    "inputs": [
        ((1, 10, 128), torch.float32),
        ((128, 128), torch.float32),
        ((128,), torch.float32),
    ],
    "outputs": [
        ((1, 10, 128), torch.float32),
    ]
}

