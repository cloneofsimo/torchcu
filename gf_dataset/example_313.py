
import torch
import torch.nn.functional as F

def torch_gelu_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies the Gaussian Error Linear Unit (GELU) activation function.
    """
    return F.gelu(input_tensor)

function_signature = {
    "name": "torch_gelu_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
