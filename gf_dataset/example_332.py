
import torch
import torch.nn.functional as F

def torch_conv2d_fp16(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int = 1, padding: int = 0) -> torch.Tensor:
    """
    Performs a 2D convolution with FP16 precision and returns the result in FP16.
    """
    input_fp16 = input_tensor.to(torch.float16)
    weight_fp16 = weight.to(torch.float16)
    bias_fp16 = bias.to(torch.float16)
    output_fp16 = F.conv2d(input_fp16, weight_fp16, bias_fp16, stride=stride, padding=padding)
    return output_fp16.to(torch.float16)


function_signature = {
    "name": "torch_conv2d_fp16",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),
        ((32, 3, 3, 3), torch.float32),
        ((32,), torch.float32),
    ],
    "outputs": [
        ((1, 32, 112, 112), torch.float16),
    ]
}
