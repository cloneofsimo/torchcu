
import torch
import torch.nn.functional as F
from torch.nn.functional import conv2d
from torch.fft import fft2, ifft2
from torch.nn.functional import relu

def torch_conv2d_fft_triplet_loss_scaled_dot_product_attention_function(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs convolution using FFT, computes triplet loss, and applies scaled dot-product attention with masking.
    """
    # Convolution using FFT
    query = conv2d(query, weight, bias)
    query_fft = fft2(query)
    key_fft = fft2(key)
    value_fft = fft2(value)
    output_fft = query_fft * key_fft
    output = ifft2(output_fft)

    # Triplet Loss
    anchor = output[:1]
    positive = output[1:2]
    negative = output[2:]
    triplet_loss = F.triplet_margin_loss(anchor, positive, negative, margin=1.0)

    # Scaled Dot-Product Attention
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.shape[-1] ** 0.5)
    attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(attention_scores, dim=-1)
    output = torch.matmul(attention_weights, value)

    return output, triplet_loss

function_signature = {
    "name": "torch_conv2d_fft_triplet_loss_scaled_dot_product_attention_function",
    "inputs": [
        ((16, 128, 28, 28), torch.float32),
        ((16, 128, 28, 28), torch.float32),
        ((16, 128, 28, 28), torch.float32),
        ((16, 28, 28), torch.bool),
        ((3, 3, 128, 128), torch.float32),
        ((128), torch.float32)
    ],
    "outputs": [
        ((16, 128, 28, 28), torch.float32),
        ((1), torch.float32),
    ]
}
