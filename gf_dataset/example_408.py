
import torch
import torch.nn.functional as F
from cutlass import *

def torch_feature_mixing_function(input_tensor: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor) -> torch.Tensor:
    """
    Mix features by performing two matrix multiplications and applying ReLU.
    """
    output1 = torch.matmul(input_tensor, weight1.t())
    output2 = torch.matmul(input_tensor, weight2.t())
    output = F.relu(output1 + output2)
    return output

function_signature = {
    "name": "torch_feature_mixing_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
