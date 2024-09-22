
import torch
import torch.fft

def pitch_correction_with_rfft(audio: torch.Tensor, pitch_shift: float, sample_rate: int) -> torch.Tensor:
    """
    Performs pitch correction on an audio signal using the real-valued fast Fourier transform (rfft).

    Args:
        audio (torch.Tensor): The audio signal as a 1D tensor of float32 values.
        pitch_shift (float): The pitch shift factor. A value of 1.0 represents no pitch shift.
        sample_rate (int): The sample rate of the audio signal in Hz.

    Returns:
        torch.Tensor: The pitch-corrected audio signal as a 1D tensor of float32 values.
    """

    # Convert audio to bfloat16 for faster processing
    audio_bf16 = audio.to(torch.bfloat16)

    # Calculate the FFT of the audio signal
    fft_audio = torch.fft.rfft(audio_bf16)

    # Determine the frequency bins corresponding to the pitch shift
    num_bins = fft_audio.shape[-1]
    frequency_bins = torch.arange(num_bins) / num_bins * sample_rate

    # Shift the frequency bins by the specified pitch shift factor
    shifted_frequency_bins = frequency_bins * pitch_shift
    shifted_fft_audio = torch.zeros_like(fft_audio)
    shifted_fft_audio[..., :num_bins] = torch.fft.rfft(audio_bf16[..., :num_bins], n=num_bins)

    # Apply the softplus activation to the shifted FFT
    shifted_fft_audio = torch.nn.functional.softplus(shifted_fft_audio)

    # Perform the inverse FFT to obtain the pitch-corrected audio signal
    corrected_audio = torch.fft.irfft(shifted_fft_audio)

    # Convert the corrected audio back to float32
    corrected_audio = corrected_audio.to(torch.float32)

    return corrected_audio

function_signature = {
    "name": "pitch_correction_with_rfft",
    "inputs": [
        ((1024,), torch.float32),  # Audio signal
        (1.0, torch.float32),       # Pitch shift factor
        (44100, torch.int32)        # Sample rate
    ],
    "outputs": [
        ((1024,), torch.float32)  # Corrected audio
    ]
}

