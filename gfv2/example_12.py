
import torch
import torch.fft

def stft_and_power_spectrum(signal: torch.Tensor, 
                            window: torch.Tensor, 
                            n_fft: int, 
                            hop_length: int, 
                            win_length: int) -> torch.Tensor:
    """
    Computes the Short-Time Fourier Transform (STFT) and power spectrum of a signal.

    Args:
        signal: The input signal tensor.
        window: The window function tensor.
        n_fft: The length of the FFT.
        hop_length: The hop length between frames.
        win_length: The length of the window function.

    Returns:
        The power spectrum of the signal.
    """
    # Perform STFT
    stft_matrix = torch.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)

    # Calculate power spectrum
    power_spectrum = stft_matrix.abs() ** 2

    return power_spectrum

function_signature = {
    "name": "stft_and_power_spectrum",
    "inputs": [
        ((1024,), torch.float32),
        ((256,), torch.float32),
        (1024,),
        (256,),
        (256,)
    ],
    "outputs": [
        ((5, 513), torch.float32)
    ]
}
