
import torch
from torch import nn
import torch.nn.functional as F

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        # Partition into windows
        x = x.view(B, self.window_size, self.window_size, C)  # B, Wh, Ww, C
        x = x.permute(0, 3, 1, 2)  # B, C, Wh, Ww

        # Query, Key, Value
        qkv = self.qkv(x).reshape(B, C, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            nW = self.window_size * self.window_size
            attn = attn.view(B, -1, self.num_heads, nW, nW) + mask.view(B, 1, 1, nW, nW)
            attn = attn.view(B, -1, self.num_heads, nW, nW)
        attn = self.attn(attn)

        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, C, self.window_size, self.window_size)
        x = x.permute(0, 2, 3, 1).view(B, N, C)  # B, N, C
        return x


def attention_with_quantization_function(input_tensor: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Perform a window attention calculation with gradient quantization.
    """
    # Gradient quantization
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        input_tensor = torch.quantize_per_tensor(input_tensor, scale=1.0, zero_point=0, dtype=torch.qint8)
        weight = torch.quantize_per_tensor(weight, scale=1.0, zero_point=0, dtype=torch.qint8)
    
    # Window attention
    attn = WindowAttention(dim=input_tensor.size(-1), num_heads=8, window_size=8)
    output = attn(input_tensor, mask)
    
    # Gradient dequantization
    output = output.dequantize()

    # BMM (using bfloat16 to reduce precision and increase speed)
    output = torch.bmm(output, weight.to(torch.bfloat16)).to(torch.float32)
    
    return output

function_signature = {
    "name": "attention_with_quantization_function",
    "inputs": [
        ((8, 128, 32), torch.float32),
        ((32, 32), torch.float32),
        ((8, 1, 8, 8), torch.bool),  # Optional mask tensor
    ],
    "outputs": [
        ((8, 128, 32), torch.float32)
    ]
}
