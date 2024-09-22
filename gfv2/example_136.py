
import torch
import torch.fft
import torch.nn.functional as F

def complex_conv2d_fft_fp16(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a 2D convolution using FFT for complex-valued inputs and weights, with FP16 precision for efficiency.
    """
    # Convert to FP16 for efficiency
    input_tensor = input_tensor.to(torch.float16)
    weight = weight.to(torch.float16)
    bias = bias.to(torch.float16)

    # Perform inverse DFT on the input tensor
    input_tensor_fft = torch.fft.irfft2(input_tensor, dim=(-2, -1))

    # Perform inverse DFT on the weight tensor
    weight_fft = torch.fft.irfft2(weight, dim=(-2, -1))

    # Perform convolution in the frequency domain
    output_fft = input_tensor_fft * weight_fft

    # Perform forward DFT to get the output in the spatial domain
    output = torch.fft.rfft2(output_fft, dim=(-2, -1))

    # Add the bias
    output += bias

    # Apply softplus activation
    output = F.softplus(output)

    # Return the output in FP32
    return output.to(torch.float32)

function_signature = {
    "name": "complex_conv2d_fft_fp16",
    "inputs": [
        ((4, 2, 16, 16), torch.complex64),
        ((2, 2, 8, 8), torch.complex64),
        ((2,), torch.float32)
    ],
    "outputs": [
        ((4, 2, 16, 16), torch.float32),
    ]
}
