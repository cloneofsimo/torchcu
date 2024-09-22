
import torch
from torch.nn.functional import interpolate

def torch_pitch_correction_cutout_int8(input_tensor: torch.Tensor, target_sr: float, cutout_size: int) -> torch.Tensor:
    """
    Performs pitch correction and cutout on an audio tensor.
    """
    # Pitch Correction
    input_tensor = torch.nn.functional.interpolate(input_tensor, size=int(input_tensor.shape[1] * target_sr / 16000), mode='linear', align_corners=False)

    # Cutout
    if cutout_size > 0:
        start_idx = torch.randint(0, input_tensor.shape[1] - cutout_size, (1,))
        input_tensor[:, start_idx : start_idx + cutout_size] = 0.0

    # Convert to int8
    input_tensor = input_tensor.to(torch.int8)

    return input_tensor

function_signature = {
    "name": "torch_pitch_correction_cutout_int8",
    "inputs": [
        ((1, 16000), torch.float32),
        (float,),
        (int,),
    ],
    "outputs": [
        ((1, 16000), torch.int8),
    ]
}
