
import torch
import torch.nn.functional as F
from torch.fft import rfft, irfft

def audio_decompression_fft_maxpool_cu(input_tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Decompresses audio data using FFT, applies max pooling, and returns the compressed representation.
    """
    # Convert to complex tensor for FFT
    complex_tensor = input_tensor.to(torch.complex64)
    
    # Apply FFT
    fft_output = rfft(complex_tensor, dim=1)

    # Max pooling in frequency domain
    max_pool_output = F.adaptive_max_pool1d(fft_output.real, output_size=kernel_size)

    # Reconstruct audio from max-pooled FFT output
    reconstructed_signal = irfft(max_pool_output.to(torch.complex64), dim=1)
    
    # Return the real part of the reconstructed signal
    return reconstructed_signal.real.to(torch.float32)

function_signature = {
    "name": "audio_decompression_fft_maxpool_cu",
    "inputs": [
        ((1, 1024), torch.float32),
        (2, )
    ],
    "outputs": [
        ((1, 2), torch.float32),
    ]
}
