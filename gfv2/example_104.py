
import torch

def harmonic_percussive_separation_bf16(audio_tensor: torch.Tensor, window_size: int, hop_length: int) -> torch.Tensor:
    """
    Performs harmonic-percussive separation using a simple time-domain approach with bfloat16 precision.

    Args:
        audio_tensor (torch.Tensor): Input audio tensor of shape (batch_size, num_channels, num_samples).
        window_size (int): Size of the sliding window in samples.
        hop_length (int): Hop length in samples.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_channels, num_samples) containing the separated harmonic component.
    """

    # Convert to bfloat16
    audio_bf16 = audio_tensor.to(torch.bfloat16)

    # Apply a simple low-pass filter (moving average)
    harmonic_bf16 = torch.nn.functional.avg_pool1d(audio_bf16, kernel_size=window_size, stride=hop_length)

    # Scale the harmonic component to match the original amplitude
    harmonic_bf16 = harmonic_bf16 * (window_size / hop_length)

    # Apply ReLU to remove negative values (ensure only harmonic components remain)
    harmonic_bf16 = torch.nn.functional.rrelu(harmonic_bf16, 0.1)

    # Convert back to float32
    harmonic_fp32 = harmonic_bf16.to(torch.float32)

    return harmonic_fp32

function_signature = {
    "name": "harmonic_percussive_separation_bf16",
    "inputs": [
        ((1, 1, 1024), torch.float32),
        (1024, ), torch.int32
        (256, ), torch.int32
    ],
    "outputs": [
        ((1, 1, 1024), torch.float32),
    ]
}
