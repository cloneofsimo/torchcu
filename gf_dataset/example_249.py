
import torch
import torch.nn.functional as F
from torch.nn import MultiLabelMarginLoss

def audio_resynthesis_loss(predicted_audio: torch.Tensor, target_audio: torch.Tensor, 
                           mel_spectrogram: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
    """
    Calculates the audio resynthesis loss, combining KL divergence for the mel-spectrogram and 
    multi-label margin loss for the audio waveform.

    Args:
        predicted_audio (torch.Tensor): Predicted audio waveform.
        target_audio (torch.Tensor): Target audio waveform.
        mel_spectrogram (torch.Tensor): Mel-spectrogram of the target audio.
        noise_level (float, optional): Noise level for the multi-label margin loss. Defaults to 0.01.

    Returns:
        torch.Tensor: The combined loss.
    """
    # KL divergence loss for mel-spectrogram
    predicted_mel = torch.stft(predicted_audio, n_fft=1024, hop_length=256, win_length=512)
    predicted_mel = torch.abs(predicted_mel) ** 2
    predicted_mel = torch.log(predicted_mel + 1e-8)
    kl_loss = F.kl_div(predicted_mel, mel_spectrogram, reduction='batchmean')

    # Multi-label margin loss for the audio waveform
    margin_loss = MultiLabelMarginLoss()(predicted_audio, target_audio)

    # Combine the losses
    total_loss = kl_loss + noise_level * margin_loss
    return total_loss

function_signature = {
    "name": "audio_resynthesis_loss",
    "inputs": [
        ((1, 16000), torch.float32),
        ((1, 16000), torch.float32),
        ((1, 128, 257), torch.float32),
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
