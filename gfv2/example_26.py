
import torch
import torch.nn.functional as F
from cutlass import *

def identity_maxpool_fp32_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs an identity operation followed by a 1D max pooling.
    """
    output = input_tensor
    output = F.max_pool1d(output, kernel_size=2, stride=2)
    return output

function_signature = {
    "name": "identity_maxpool_fp32_function",
    "inputs": [
        ((16, 1, 32), torch.float32),
    ],
    "outputs": [
        ((16, 1, 16), torch.float32),
    ]
}
