
import torch
from torch.utils.checkpoint import checkpoint

def int8_gradient_checkpointing(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a simple linear transformation using int8 quantization with gradient checkpointing.
    """
    input_int8 = input_tensor.to(torch.int8)
    weight_int8 = weight.to(torch.int8)
    output = checkpoint(torch.matmul, input_int8, weight_int8.t())
    return output.to(torch.float32)

function_signature = {
    "name": "int8_gradient_checkpointing",
    "inputs": [
        ((1, 16), torch.float32),
        ((16, 16), torch.float32)
    ],
    "outputs": [
        ((1, 16), torch.float32),
    ]
}
