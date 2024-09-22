
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def torch_vocoder_bfloat16_function(
    mel_spectrogram: torch.Tensor,
    decoder_states: torch.Tensor,
    vocoder_weights: torch.Tensor,
    vocoder_biases: torch.Tensor
) -> torch.Tensor:
    """
    Performs vocoding using a bfloat16 linear layer and pixel unshuffle.

    Args:
        mel_spectrogram: Mel-spectrogram input, shape (batch_size, mel_channels, mel_length).
        decoder_states: Decoder states, shape (batch_size, decoder_states_dim).
        vocoder_weights: Vocoder weights, shape (vocoder_output_dim, decoder_states_dim + mel_channels).
        vocoder_biases: Vocoder biases, shape (vocoder_output_dim).

    Returns:
        vocoded_audio: Vocoded audio, shape (batch_size, vocoder_output_dim, mel_length).
    """

    with autocast():
        # Concatenate decoder states and mel-spectrogram
        combined_input = torch.cat([decoder_states, mel_spectrogram.transpose(1, 2)], dim=2)
        # Linear transformation
        vocoded_audio = F.linear(combined_input, vocoder_weights, bias=vocoder_biases)
        # Pixel unshuffle
        vocoded_audio = F.pixel_unshuffle(vocoded_audio, downscale_factor=2)

    return vocoded_audio

function_signature = {
    "name": "torch_vocoder_bfloat16_function",
    "inputs": [
        ((1, 80, 256), torch.float32),
        ((1, 256), torch.float32),
        ((128, 256 + 80), torch.float32),
        ((128,), torch.float32)
    ],
    "outputs": [
        ((1, 128, 256), torch.float32),
    ]
}
