
import torch
import torch.fft

def cross_fade_fft_shift(audio1: torch.Tensor, audio2: torch.Tensor, crossfade_start: float, crossfade_duration: float) -> torch.Tensor:
    """
    Performs a crossfade between two audio signals in the frequency domain, applying fftshift for better visualization.
    """
    assert audio1.shape == audio2.shape, "Audio signals must have the same shape."
    
    # Calculate the sample indices for the crossfade
    start_sample = int(crossfade_start * audio1.shape[-1])
    end_sample = int((crossfade_start + crossfade_duration) * audio1.shape[-1])

    # Apply FFT to both audio signals
    fft1 = torch.fft.fft(audio1)
    fft2 = torch.fft.fft(audio2)

    # Perform crossfade in the frequency domain
    fft1[..., start_sample:end_sample] = (
        fft1[..., start_sample:end_sample] * (1 - (end_sample - start_sample) / crossfade_duration) 
        + fft2[..., start_sample:end_sample] * ((end_sample - start_sample) / crossfade_duration)
    )

    # Apply fftshift for better visualization
    fft1 = torch.fft.fftshift(fft1, dim=-1)

    # Perform inverse FFT and return the crossfaded audio signal
    return torch.fft.ifft(torch.fft.ifftshift(fft1, dim=-1)).real

function_signature = {
    "name": "cross_fade_fft_shift",
    "inputs": [
        ((1024,), torch.float32),
        ((1024,), torch.float32),
        (torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((1024,), torch.float32),
    ]
}
