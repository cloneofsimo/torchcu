
import torch
import torchaudio

def torch_pitch_correction_function(input_tensor: torch.Tensor, semitones: float) -> torch.Tensor:
    """
    Applies pitch correction to an audio signal using the TorchAudio's pitch_shift function.
    """
    return torchaudio.functional.pitch_shift(input_tensor, 16000, semitones)

function_signature = {
    "name": "torch_pitch_correction_function",
    "inputs": [
        ((1, 16000), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((1, 16000), torch.float32)
    ]
}
