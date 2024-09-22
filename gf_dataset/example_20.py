
import torch
import torch.nn.functional as F

def pitch_correction_flatten_int8(input_tensor: torch.Tensor, pitch_shift: float, scale: float) -> torch.Tensor:
    """
    Performs pitch correction, flattens the tensor, and returns the result as int8.
    """
    # Pitch correction
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
    input_tensor = torch.nn.functional.interpolate(input_tensor, scale_factor=pitch_shift, mode='linear', align_corners=False)

    # Flatten the tensor
    flattened_tensor = input_tensor.flatten()

    # Quantize to int8
    quantized_tensor = (flattened_tensor * scale).to(torch.int8)

    return quantized_tensor

function_signature = {
    "name": "pitch_correction_flatten_int8",
    "inputs": [
        ((1, 16000), torch.float32),
        (torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((16000,), torch.int8),
    ]
}
