
import torch
import torch.nn.functional as F

def audio_decompress_torch(input_tensor: torch.Tensor, scale: float, offset: float) -> torch.Tensor:
    """
    Decompresses an audio tensor, scaling and offsetting it.
    """
    input_tensor = input_tensor.to(torch.float32)
    output = input_tensor * scale + offset
    output = F.relu(output)  # Apply ReLU for non-negative values
    return output

function_signature = {
    "name": "audio_decompress_torch",
    "inputs": [
        ((1, 16000), torch.float32),
        ((), torch.float32),
        ((), torch.float32)
    ],
    "outputs": [
        ((1, 16000), torch.float32),
    ]
}
