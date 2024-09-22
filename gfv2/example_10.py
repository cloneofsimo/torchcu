
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

def coord_attention_bf16_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    CoordAttention with bfloat16 precision. 
    """
    B, C, H, W = input_tensor.size()
    input_bf16 = input_tensor.to(torch.bfloat16)

    # Generate coordinate embeddings
    x_range = torch.arange(W, dtype=torch.float32, device=input_tensor.device)
    y_range = torch.arange(H, dtype=torch.float32, device=input_tensor.device)
    x_embed = x_range.unsqueeze(1).repeat(1, H).view(1, 1, H, W)
    y_embed = y_range.unsqueeze(0).repeat(W, 1).view(1, 1, H, W)

    # Concatenate coordinate embeddings with input
    input_embed = torch.cat((input_bf16, x_embed, y_embed), dim=1)
    
    # Apply convolution
    output_bf16 = F.conv2d(input_embed, weight, padding="same", bias=None).to(torch.bfloat16)

    # Apply cosine similarity for attention
    attention = F.cosine_similarity(input_bf16, output_bf16, dim=1, eps=1e-6)
    attention = attention.unsqueeze(1)

    # Multiply attention weights with input
    output = (attention * input_tensor).to(torch.float32)

    return output

function_signature = {
    "name": "coord_attention_bf16_function",
    "inputs": [
        ((2, 64, 16, 16), torch.float32),
        ((66, 3, 3), torch.float32)
    ],
    "outputs": [
        ((2, 64, 16, 16), torch.float32)
    ]
}
