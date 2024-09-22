
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def torch_spectrogram_bandwidth_fn(audio: torch.Tensor, sample_rate: int, window_size: int, hop_size: int) -> torch.Tensor:
    """
    Calculates the spectral bandwidth of a spectrogram.

    Args:
        audio (torch.Tensor): Audio signal tensor of shape (batch, time).
        sample_rate (int): Sample rate of the audio signal.
        window_size (int): Size of the window in samples.
        hop_size (int): Hop size in samples.

    Returns:
        torch.Tensor: Spectral bandwidth of the spectrogram, shape (batch, time).
    """
    with autocast():
        spectrogram = torch.stft(
            audio,
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=torch.hann_window(window_size),
            center=True,
            pad_mode="reflect",
            return_complex=False,
        )
        spectrogram = spectrogram.pow(2)
        spectrogram_sum = spectrogram.sum(dim=1, keepdim=True)
        
        # Calculate mean frequency
        frequency_bins = torch.arange(spectrogram.shape[1], device=spectrogram.device) * sample_rate / window_size
        mean_frequency = (spectrogram * frequency_bins).sum(dim=1, keepdim=True) / spectrogram_sum
        
        # Calculate squared deviation from mean frequency
        squared_deviation = (frequency_bins - mean_frequency)**2
        
        # Calculate spectral bandwidth
        bandwidth = torch.sqrt((spectrogram * squared_deviation).sum(dim=1, keepdim=True) / spectrogram_sum)
    return bandwidth.to(torch.float32)

function_signature = {
    "name": "torch_spectrogram_bandwidth_fn",
    "inputs": [
        ((1, 16000), torch.float32),
        ((), torch.int32),
        ((), torch.int32),
        ((), torch.int32),
    ],
    "outputs": [
        ((1, 801), torch.float32)
    ]
}
