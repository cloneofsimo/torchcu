
import torch
import torch.nn.functional as F

def torch_conv_std_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a convolution with a specified kernel and calculates the standard deviation along the channel dimension.
    """
    output = F.conv2d(input_tensor.float(), weight.float(), bias.float(), stride=1, padding=1)
    std_output = torch.std(output, dim=1, keepdim=True)
    return std_output.half()

function_signature = {
    "name": "torch_conv_std_function",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),
        ((3, 3, 3, 3), torch.float32),
        ((3,), torch.float32)
    ],
    "outputs": [
        ((1, 1, 224, 224), torch.float16),
    ]
}

