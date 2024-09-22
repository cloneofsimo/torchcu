
import torch
import torch.nn.functional as F

def torch_dynamic_conv_function(input_tensor: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int, groups: int, inplace: bool=False) -> torch.Tensor:
    """
    Performs a dynamic convolution with optional inplace operation.
    """
    output = F.conv2d(input_tensor, weights, bias, kernel_size, stride, padding, dilation, groups)
    if inplace:
        input_tensor.data.copy_(output.data)
        return input_tensor
    return output

function_signature = {
    "name": "torch_dynamic_conv_function",
    "inputs": [
        ((1, 16, 28, 28), torch.float32),
        ((16, 16, 3, 3), torch.float32),
        ((16,), torch.float32),
        (3,),
        (2,),
        (1,),
        (2,),
        (16,),
        (False,),
    ],
    "outputs": [
        ((1, 16, 13, 13), torch.float32),
    ]
}
