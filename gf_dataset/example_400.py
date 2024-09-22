
import torch
import torch.nn.functional as F
from torch.cuda import cutlass

def torch_grouped_conv_frobenius_norm(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a grouped convolution, applies Swish activation, and then calculates the Frobenius norm of the output.
    """
    output = F.conv2d(input_tensor, weight, bias=bias, groups=groups)
    output = F.swish(output)
    frobenius_norm = torch.linalg.norm(output, ord='fro')
    return output, frobenius_norm

function_signature = {
    "name": "torch_grouped_conv_frobenius_norm",
    "inputs": [
        ((1, 64, 112, 112), torch.float32),
        ((32, 64, 3, 3), torch.float32),
        ((32,), torch.float32),
        (32,),
    ],
    "outputs": [
        ((1, 32, 112, 112), torch.float32),
        ((1,), torch.float32)
    ]
}
