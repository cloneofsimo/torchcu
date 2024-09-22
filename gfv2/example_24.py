
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def audio_resynthesis_bf16(
    spectrogram: torch.Tensor, 
    mel_basis: torch.Tensor, 
    stft_params: dict
) -> torch.Tensor:
    """
    Resynthesizes audio from a spectrogram using bfloat16 for optimized performance.

    Args:
        spectrogram (torch.Tensor): The input spectrogram tensor of shape (batch_size, n_fft // 2 + 1, frames).
        mel_basis (torch.Tensor): The mel basis matrix of shape (n_mels, n_fft // 2 + 1).
        stft_params (dict): A dictionary containing STFT parameters:
            - n_fft (int): The FFT size.
            - hop_length (int): The hop length.
            - window (str or torch.Tensor): The window function (e.g., "hann").

    Returns:
        torch.Tensor: The resynthesized audio tensor of shape (batch_size, frames * hop_length).
    """
    with autocast():
        # Convert spectrogram to mel-spectrogram
        mel_spectrogram = torch.matmul(mel_basis, spectrogram)

        # Invert mel-spectrogram to spectrogram (using inverse mel basis)
        spectrogram = torch.matmul(torch.linalg.pinv(mel_basis), mel_spectrogram)

        # Perform inverse STFT
        audio = torch.istft(spectrogram,
                           n_fft=stft_params["n_fft"],
                           hop_length=stft_params["hop_length"],
                           window=stft_params["window"])
    return audio.to(torch.float32)

function_signature = {
    "name": "audio_resynthesis_bf16",
    "inputs": [
        ((16, 257, 256), torch.float32),
        ((80, 257), torch.float32),
        {"n_fft": 1024, "hop_length": 256, "window": "hann"}
    ],
    "outputs": [
        ((16, 65536), torch.float32)
    ]
}
