
import torch
import torch.nn.functional as F
from cutlass import *

def torch_bernoulli_bfloat16_function(input_tensor: torch.Tensor, prob: float) -> torch.Tensor:
    """
    Applies a Bernoulli distribution to an input tensor with given probability, 
    using bfloat16 for intermediate calculations and returning a float32 tensor.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output_bf16 = torch.bernoulli(input_bf16, p=prob)
    return output_bf16.to(torch.float32)

function_signature = {
    "name": "torch_bernoulli_bfloat16_function",
    "inputs": [
        ((4, 4), torch.float32),
        (float, torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
