
import torch
import torch.nn.functional as F

def torch_flatten_relu(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Flatten the input tensor and apply ReLU activation.
    """
    flattened = torch.flatten(input_tensor, start_dim=1)
    return F.relu(flattened)

function_signature = {
    "name": "torch_flatten_relu",
    "inputs": [
        ((2, 3, 4), torch.float32),
    ],
    "outputs": [
        ((2, 12), torch.float32)
    ]
}
