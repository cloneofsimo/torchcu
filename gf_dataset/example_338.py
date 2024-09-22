
import torch
import torch.nn.functional as F
from torch.cuda import cudnn

def torch_conv2d_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a 2D convolution with specified weight and bias, followed by ReLU activation. 
    """
    output = F.conv2d(input_tensor, weight, bias=bias, padding=1)
    return F.relu(output)

function_signature = {
    "name": "torch_conv2d_function",
    "inputs": [
        ((1, 3, 32, 32), torch.float32),
        ((3, 3, 3, 3), torch.float32),
        ((3,), torch.float32),
    ],
    "outputs": [
        ((1, 3, 32, 32), torch.float32),
    ]
}

