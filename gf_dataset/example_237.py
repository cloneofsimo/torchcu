
import torch
import torch.nn.functional as F
from cutlass import *

def torch_prelu_noise_bfloat16(input_tensor: torch.Tensor, weight: torch.Tensor, noise_scale: float) -> torch.Tensor:
    """
    Applies PReLU activation with noise injection in bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    output = F.prelu(input_bf16, weight_bf16)
    output = output + torch.randn_like(output, dtype=torch.bfloat16) * noise_scale
    return output.to(torch.float16)

function_signature = {
    "name": "torch_prelu_noise_bfloat16",
    "inputs": [
        ((4, 4), torch.float32),
        ((1,), torch.float32),
        ((), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float16),
    ]
}
