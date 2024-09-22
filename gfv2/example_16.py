
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def torch_lightweight_conv_hardshrink_masked_attention_fp16_int8(
    input_tensor: torch.Tensor, 
    weight: torch.Tensor, 
    bias: torch.Tensor, 
    mask: torch.Tensor, 
    threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a lightweight convolution, applies hard shrink activation, 
    and then applies masked attention using int8 precision.
    """
    # Convert input and weight to fp16
    input_tensor_fp16 = input_tensor.to(torch.float16)
    weight_fp16 = weight.to(torch.float16)

    # Perform lightweight convolution using int8
    output_int8 = F.conv2d(input_tensor_fp16, weight_fp16, bias=bias, groups=1, padding=1).to(torch.int8)

    # Apply hard shrink activation
    output_fp16 = F.hardshrink(output_int8.to(torch.float16), lambd=threshold)

    # Apply masked attention
    output_masked = output_fp16 * mask

    # Return both output tensor and the masked output tensor
    return output_masked, output_fp16

function_signature = {
    "name": "torch_lightweight_conv_hardshrink_masked_attention_fp16_int8",
    "inputs": [
        ((1, 16, 32, 32), torch.float32),
        ((8, 16, 3, 3), torch.float32),
        ((8,), torch.float32),
        ((1, 1, 32, 32), torch.float32),
        (0.5, torch.float32),
    ],
    "outputs": [
        ((1, 8, 32, 32), torch.float16),
        ((1, 8, 32, 32), torch.float16)
    ]
}
