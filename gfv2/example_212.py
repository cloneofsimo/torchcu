
import torch

def grouped_conv_eq_fp16(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a grouped convolution with equal weights and bias, returning the result in FP16.
    """
    # Convert to FP16
    input_tensor_fp16 = input_tensor.to(torch.float16)
    weight_fp16 = weight.to(torch.float16)
    bias_fp16 = bias.to(torch.float16)

    # Grouped convolution
    output_fp16 = torch.nn.functional.conv2d(input_tensor_fp16, weight_fp16, bias_fp16, groups=input_tensor.shape[1])
    return output_fp16

function_signature = {
    "name": "grouped_conv_eq_fp16",
    "inputs": [
        ((1, 16, 16, 16), torch.float32),
        ((16, 1, 3, 3), torch.float32),
        ((16,), torch.float32)
    ],
    "outputs": [
        ((1, 16, 14, 14), torch.float16),
    ]
}
