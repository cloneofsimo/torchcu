
import torch
import torch.nn.functional as F

def torch_grouped_conv2d_int8(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Performs a 2D grouped convolution with int8 weights and bias.
    """
    input_tensor = input_tensor.to(torch.int8)
    weight = weight.to(torch.int8)
    bias = bias.to(torch.int8)
    output = F.conv2d(input_tensor, weight, bias, groups=groups, padding=1)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_grouped_conv2d_int8",
    "inputs": [
        ((1, 3, 32, 32), torch.float32), 
        ((32, 3, 3, 3), torch.float32),
        ((32,), torch.float32),
        (32,)
    ],
    "outputs": [
        ((1, 32, 32, 32), torch.float32)
    ]
}
