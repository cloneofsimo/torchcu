
import torch
import torch.nn.functional as F
from torch.nn import Conv2d

def torch_dynamic_conv_int8_bfloat16_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Performs a dynamic convolution with int8 quantization and bfloat16 accumulation, followed by a sigmoid activation. 
    """
    # Dynamic convolution
    input_tensor = input_tensor.to(torch.bfloat16)
    weight = weight.to(torch.int8)
    bias = bias.to(torch.bfloat16)
    output = F.conv2d(input_tensor, weight, bias, kernel_size=kernel_size, padding=kernel_size // 2, groups=1)
    # Apply sigmoid activation
    output = torch.sigmoid(output.to(torch.float32))
    return output

function_signature = {
    "name": "torch_dynamic_conv_int8_bfloat16_function",
    "inputs": [
        ((1, 1, 16, 16), torch.float32),
        ((1, 1, 3, 3), torch.int8),
        ((1,), torch.float32),
        (3,)
    ],
    "outputs": [
        ((1, 1, 16, 16), torch.float32),
    ]
}
