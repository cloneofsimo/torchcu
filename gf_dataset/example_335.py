
import torch
import torch.nn.functional as F

def audio_upsampling_with_softmax_temperature(audio_tensor: torch.Tensor, upsampling_factor: int, temperature: float) -> torch.Tensor:
    """
    Upsamples audio tensor using linear interpolation, then applies softmax with temperature scaling.
    """
    upsampled_audio = F.interpolate(audio_tensor, scale_factor=upsampling_factor, mode='linear')
    softmax_output = F.softmax(upsampled_audio / temperature, dim=-1)
    return softmax_output

function_signature = {
    "name": "audio_upsampling_with_softmax_temperature",
    "inputs": [
        ((100, 16000), torch.float32),
        (int, None),
        (float, None),
    ],
    "outputs": [
        ((100, upsampling_factor * 16000), torch.float32)
    ]
}
