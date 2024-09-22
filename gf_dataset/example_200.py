
import torch
import torch.nn.functional as F
from torch.cuda import amp

def torch_lightweight_conv_fp16_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                                         stride: int = 1, padding: int = 1) -> torch.Tensor:
    """
    Performs a lightweight convolution using FP16, optionally with weight standardization and bias.
    """
    with amp.autocast(dtype=torch.float16):
        output = F.conv2d(input_tensor.half(), weight.half(), bias.half(), stride=stride, padding=padding)
        output = output.float()  # Convert back to FP32
        return output

function_signature = {
    "name": "torch_lightweight_conv_fp16_function",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),
        ((3, 3, 3, 3), torch.float32),
        ((3,), torch.float32),
    ],
    "outputs": [
        ((1, 3, 224, 224), torch.float32),
    ]
}
