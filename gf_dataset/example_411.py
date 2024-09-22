
import torch
import torch.nn.functional as F
from torch.nn import AdaptiveMaxPool1d

def torch_function(input_tensor: torch.Tensor, weights: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Performs a sequence of operations, including rotary positional encoding, attention, and pooling, using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weights_bf16 = weights.to(torch.bfloat16)

    # Rotary positional encoding
    freqs = 10000 ** (torch.arange(0, input_tensor.shape[-1], 2, device=input_tensor.device) / input_tensor.shape[-1])
    pos_sin = torch.sin(freqs * input_bf16)
    pos_cos = torch.cos(freqs * input_bf16)
    rotary_input = torch.stack([pos_cos, pos_sin], dim=-1)

    # Attention (using einsum_outer)
    attention = torch.einsum('b i d, b j d -> b i j', input_bf16, rotary_input)

    # Adaptive max pooling
    pool = AdaptiveMaxPool1d(1)
    output = pool(attention).squeeze(-1)

    return output.to(torch.float32)

function_signature = {
    "name": "torch_function",
    "inputs": [
        ((16, 128, 512), torch.float32),
        ((128, 512), torch.float32),
        (512, torch.int64),
    ],
    "outputs": [
        ((16, 128), torch.float32),
    ]
}
