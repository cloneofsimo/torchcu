
import torch
import torch.nn.functional as F

def torch_mel_spectrogram_function(audio_tensor: torch.Tensor, sample_rate: int, n_fft: int, hop_length: int, win_length: int, n_mels: int, f_min: int, f_max: int) -> torch.Tensor:
    """
    Calculate Mel-spectrogram from an audio tensor.
    """
    spectrogram = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    magnitude_spectrogram = torch.abs(spectrogram)
    mel_filter = torch.nn.functional.linear_to_mel(
        torch.linspace(0, sample_rate // 2, n_fft // 2 + 1),
        n_mels,
        f_min=f_min,
        f_max=f_max,
        sample_rate=sample_rate
    )
    mel_spectrogram = torch.matmul(mel_filter, magnitude_spectrogram**2)
    return torch.log1p(mel_spectrogram).to(torch.float32)

function_signature = {
    "name": "torch_mel_spectrogram_function",
    "inputs": [
        ((1, 16000), torch.float32),
        (1, torch.int32),
        (1, torch.int32),
        (1, torch.int32),
        (1, torch.int32),
        (1, torch.int32),
        (1, torch.int32),
        (1, torch.int32)
    ],
    "outputs": [
        ((1, 128, 129), torch.float32),
    ]
}
