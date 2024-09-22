
import torch
import torch.nn as nn

def hyperparameter_optimized_relu(input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Applies a ReLU activation with an optional scaling factor (alpha).

    Args:
        input_tensor (torch.Tensor): Input tensor.
        alpha (float): Scaling factor for the ReLU activation.

    Returns:
        torch.Tensor: Output tensor with applied ReLU activation and scaling.
    """
    output = torch.relu(input_tensor)
    output = output * alpha
    return output

function_signature = {
    "name": "hyperparameter_optimized_relu",
    "inputs": [
        ((4, 4), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
