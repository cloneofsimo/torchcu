
import torch
import torch.nn.functional as F
from torch.fft import rfft, irfft
import numpy as np

def torch_causal_attention_stft_gt(input_tensor: torch.Tensor, weights: torch.Tensor, gt: torch.Tensor, hop_length: int, window_length: int) -> torch.Tensor:
    """
    Applies a causal attention mechanism to the STFT of the input tensor, then
    reconstructs the signal using the inverse STFT, and calculates the L1 loss
    against the ground truth.

    Args:
        input_tensor: The input tensor of shape (batch_size, time_steps, channels).
        weights: The attention weights of shape (batch_size, num_heads, time_steps, time_steps).
        gt: The ground truth tensor of shape (batch_size, time_steps, channels).
        hop_length: The hop length for the STFT.
        window_length: The window length for the STFT.

    Returns:
        The L1 loss between the reconstructed signal and the ground truth.
    """

    # STFT
    stft_input = torch.stft(input_tensor, n_fft=window_length, hop_length=hop_length, return_complex=True)

    # Apply attention
    batch_size, num_heads, num_frames, num_bins = weights.shape
    attention_output = torch.zeros_like(stft_input)
    for i in range(num_heads):
        attention_output[:, :, :, i] = torch.matmul(weights[:, i, :, :], stft_input[:, :, :, i])

    # Inverse STFT
    reconstructed_signal = torch.istft(attention_output, n_fft=window_length, hop_length=hop_length, length=input_tensor.shape[1])

    # L1 loss
    loss = F.l1_loss(reconstructed_signal, gt)

    return loss

function_signature = {
    "name": "torch_causal_attention_stft_gt",
    "inputs": [
        ((16, 1024, 1), torch.float32),
        ((16, 8, 1024, 1024), torch.float32),
        ((16, 1024, 1), torch.float32),
        (256, torch.int32),
        (512, torch.int32)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
