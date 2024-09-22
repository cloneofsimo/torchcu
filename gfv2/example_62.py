
import torch
import torch.fft
import torch.nn.functional as F

def depthwise_separable_conv_fp16_dft_kronecker_istft(input_tensor: torch.Tensor, weight_depthwise: torch.Tensor, weight_pointwise: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int, groups: int, dft_size: int, kronecker_factor: int) -> torch.Tensor:
    """
    Performs a depthwise separable convolution followed by a DFT, Kronecker product, and inverse DFT.
    All operations are performed in fp16 for potential speed improvements.
    """

    # Depthwise convolution
    input_fp16 = input_tensor.to(torch.float16)
    weight_depthwise_fp16 = weight_depthwise.to(torch.float16)
    output_depthwise = F.conv2d(input_fp16, weight_depthwise_fp16, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # Pointwise convolution
    weight_pointwise_fp16 = weight_pointwise.to(torch.float16)
    output_pointwise = F.conv2d(output_depthwise, weight_pointwise_fp16, kernel_size=1, stride=1, padding=0)

    # DFT
    output_dft = torch.fft.fft2(output_pointwise, dim=(-2, -1))

    # Kronecker product
    output_kronecker = torch.kron(output_dft, torch.ones(kronecker_factor, kronecker_factor, dtype=torch.float16, device=input_tensor.device))

    # Inverse DFT
    output_istft = torch.fft.ifft2(output_kronecker, dim=(-2, -1))

    return output_istft.to(torch.float32)

function_signature = {
    "name": "depthwise_separable_conv_fp16_dft_kronecker_istft",
    "inputs": [
        ((4, 3, 16, 16), torch.float32),  # Input tensor
        ((3, 1, 3, 3), torch.float32),   # Depthwise weight
        ((12, 3, 1, 1), torch.float32),  # Pointwise weight
        (3, ), torch.int32,          # Kernel size
        (2, ), torch.int32,          # Stride
        (1, ), torch.int32,          # Padding
        (1, ), torch.int32,          # Dilation
        (3, ), torch.int32,          # Groups
        (16, ), torch.int32,         # DFT size
        (2, ), torch.int32           # Kronecker factor
    ],
    "outputs": [
        ((4, 12, 8, 8), torch.float32),  # Output tensor
    ]
}
