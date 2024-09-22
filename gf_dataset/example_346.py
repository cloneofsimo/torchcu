
import torch
import torch.nn.functional as F
from torch.fft import irfft, rfft
import numpy as np

def torch_causal_attention_int8_function(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Applies a causal attention mechanism with int8 precision.
    """
    # Convert to int8
    query = query.to(torch.int8)
    key = key.to(torch.int8)
    value = value.to(torch.int8)
    mask = mask.to(torch.bool)

    # Calculate attention scores
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / np.sqrt(key.size(-1))
    scores = torch.where(mask, scores, torch.finfo(torch.float32).min)  # Mask out future tokens
    attention = F.softmax(scores, dim=-1)

    # Calculate weighted sum of values
    output = torch.matmul(attention, value)

    # Convert back to float32
    return output.to(torch.float32)

function_signature = {
    "name": "torch_causal_attention_int8_function",
    "inputs": [
        ((10, 16, 64), torch.float32),
        ((10, 16, 64), torch.float32),
        ((10, 16, 64), torch.float32),
        ((10, 16, 16), torch.bool),
    ],
    "outputs": [
        ((10, 16, 64), torch.float32),
    ]
}
