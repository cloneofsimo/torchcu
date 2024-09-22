
import torch
import torch.nn.functional as F

def torch_transposed_conv1d_bf16(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                                  output_size: int, padding: int, stride: int) -> torch.Tensor:
    """
    Performs a transposed 1D convolution with bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    output = F.conv_transpose1d(input_bf16, weight_bf16, bias_bf16, stride=stride, padding=padding,
                                 output_size=output_size)
    
    # Interpolate to desired output size
    output = F.interpolate(output, size=output_size, mode='linear', align_corners=False)
    
    return output.to(torch.float16)

function_signature = {
    "name": "torch_transposed_conv1d_bf16",
    "inputs": [
        ((16, 32, 100), torch.float32),
        ((32, 16, 5), torch.float32),
        ((32,), torch.float32),
        (150,),
        (2,),
        (2,)
    ],
    "outputs": [
        ((16, 32, 150), torch.float16),
    ]
}
