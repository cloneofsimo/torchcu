
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def sparse_conv1d_fft_dropout_backward(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor, dropout_p: float) -> torch.Tensor:
    """
    Performs a sparse 1D convolution with FFT, dropout, and backward pass.

    Args:
        input_tensor: Input tensor of shape (batch_size, in_channels, seq_length).
        weight: Convolution kernel of shape (out_channels, in_channels, kernel_size).
        bias: Bias tensor of shape (out_channels).
        mask: Sparse mask tensor of shape (batch_size, in_channels, seq_length).
        dropout_p: Dropout probability.

    Returns:
        The gradients of the input tensor.
    """

    # Apply sparse mask
    masked_input = input_tensor * mask

    # 1D convolution with FFT
    output = F.conv1d(masked_input, weight, bias=bias, padding='same')

    # Dropout
    output = F.dropout(output, p=dropout_p, training=True)

    # Backward pass
    output.backward(torch.ones_like(output))

    # Return the gradients of the input tensor
    return input_tensor.grad

function_signature = {
    "name": "sparse_conv1d_fft_dropout_backward",
    "inputs": [
        ((16, 32, 128), torch.float32),
        ((64, 32, 5), torch.float32),
        ((64,), torch.float32),
        ((16, 32, 128), torch.bool),
        (0.5, None)
    ],
    "outputs": [
        ((16, 32, 128), torch.float32),
    ]
}
