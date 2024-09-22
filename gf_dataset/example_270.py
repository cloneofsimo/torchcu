
import torch
import torch.nn.functional as F

def torch_fused_gelu_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a fused GELU activation on the input tensor.
    """
    return F.gelu(input_tensor)

function_signature = {
    "name": "torch_fused_gelu_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
