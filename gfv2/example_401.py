
import torch
import torch.nn.functional as F
from torch.nn import ELU

def swin_transformer_elu_max(input_tensor: torch.Tensor, attention_weights: torch.Tensor,
                             shift_size: int, window_size: int) -> torch.Tensor:
    """
    Performs a Swin Transformer block with ELU activation and max pooling.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
        attention_weights (torch.Tensor): Attention weights of shape (B, num_heads, H/window_size, W/window_size,
                                           window_size, window_size).
        shift_size (int): Shift size for the window partitioning.
        window_size (int): Window size for the local attention.

    Returns:
        torch.Tensor: Output tensor of shape (B, C, H, W).
    """
    # Swin Transformer windowed attention
    B, C, H, W = input_tensor.size()
    input_tensor = input_tensor.view(B, C, H // window_size, window_size, W // window_size, window_size)
    input_tensor = input_tensor.permute(0, 2, 4, 1, 3, 5)
    input_tensor = input_tensor.reshape(-1, C, window_size, window_size)
    attention_weights = attention_weights.view(B, -1, window_size, window_size)
    output = torch.einsum('bchw,bmhw->bmcv', input_tensor, attention_weights)
    output = output.view(B, H // window_size, W // window_size, C, window_size, window_size)
    output = output.permute(0, 3, 1, 4, 2, 5)
    output = output.reshape(B, C, H, W)

    # Shift window
    if shift_size > 0:
        output = torch.roll(output, shifts=(-shift_size, -shift_size), dims=(2, 3))

    # ELU activation
    output = F.elu(output)

    # Max pooling
    output = F.max_pool2d(output, kernel_size=2, stride=2)

    return output

function_signature = {
    "name": "swin_transformer_elu_max",
    "inputs": [
        ((1, 128, 224, 224), torch.float32),
        ((1, 4, 14, 14, 7, 7), torch.float32),
        (1, ), torch.int32,
        (1, ), torch.int32
    ],
    "outputs": [
        ((1, 128, 112, 112), torch.float32),
    ]
}
