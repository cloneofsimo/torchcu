
import torch
from torch import nn
import torch.nn.functional as F
from cutlass import *


def torch_window_attention_function(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Window-based attention function with coordinate convolution and self-attention.
    """
    b, h, w, c = q.shape
    q = q.view(b, h, w, -1, 8)  # Reshape for coordinate convolution
    k = k.view(b, h, w, -1, 8)
    v = v.view(b, h, w, -1, 8)
    
    # Coordinate convolution
    q = coord_conv(q)
    k = coord_conv(k)
    
    # Self-attention
    attn = torch.matmul(q, k.transpose(-2, -1))
    attn = attn / (c ** 0.5)
    attn = attn.masked_fill(mask == 0, -1e9)
    attn = F.softmax(attn, dim=-1)
    
    # Weighted sum
    output = torch.matmul(attn, v)
    output = output.view(b, h, w, c)
    return output

function_signature = {
    "name": "torch_window_attention_function",
    "inputs": [
        ((4, 4, 4, 128), torch.float32),
        ((4, 4, 4, 128), torch.float32),
        ((4, 4, 4, 128), torch.float32),
        ((4, 4, 4), torch.bool)
    ],
    "outputs": [
        ((4, 4, 4, 128), torch.float32),
    ]
}

def coord_conv(x):
    """
    Coordinate convolution layer.
    """
    b, h, w, c, d = x.shape
    
    # Create coordinate grid
    coords_h = torch.arange(h, device=x.device).view(1, h, 1, 1, 1).expand(b, h, w, c, d)
    coords_w = torch.arange(w, device=x.device).view(1, 1, w, 1, 1).expand(b, h, w, c, d)
    
    # Concatenate coordinates with input
    x = torch.cat([x, coords_h, coords_w], dim=-1)
    
    # Convolution
    x = nn.Conv3d(c + 2, c, kernel_size=3, padding=1, groups=c)(x)
    
    return x

