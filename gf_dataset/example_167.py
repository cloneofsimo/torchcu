
import torch
import torch.nn.functional as F

def audio_decompression_and_loss(encoded_audio: torch.Tensor, target_audio: torch.Tensor, codec_config: list) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decompresses encoded audio using the provided codec configuration and calculates the multi-label margin loss.
    """
    # Decompress the audio
    decoded_audio = torch.nn.functional.conv_transpose2d(
        encoded_audio, codec_config[0].to(encoded_audio.dtype), stride=codec_config[1], padding=codec_config[2]
    )
    # Calculate the multi-label margin loss
    loss = F.multilabel_margin_loss(decoded_audio, target_audio)
    return decoded_audio, loss

function_signature = {
    "name": "audio_decompression_and_loss",
    "inputs": [
        ((1, 128, 128), torch.float32),
        ((1, 128, 128), torch.float32),
        ((1, 128, 128), torch.float32)  # codec_config is a list of tensors
    ],
    "outputs": [
        ((1, 128, 128), torch.float32),
        ((1,), torch.float32)
    ]
}
