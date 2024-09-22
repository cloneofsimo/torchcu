
import torch
import torch.nn.functional as F

def reflection_pad_and_relu(input_tensor: torch.Tensor, padding: int) -> torch.Tensor:
    """
    Applies reflection padding to the input tensor and then applies ReLU activation.
    """
    padded_tensor = F.pad(input_tensor, (padding, padding, padding, padding), mode='reflect')
    return F.relu(padded_tensor)

function_signature = {
    "name": "reflection_pad_and_relu",
    "inputs": [
        ((3, 3), torch.float32),
    ],
    "outputs": [
        ((3 + 2 * padding, 3 + 2 * padding), torch.float32),
    ]
}
