
import torch
import torch.nn.functional as F
from torch.fft import rfft, irfft

def torch_conv_fft_fp16(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int) -> torch.Tensor:
    """
    Performs a convolution using FFT for faster computation, working with fp16 precision. 
    """
    input_fp16 = input_tensor.to(torch.float16)
    weight_fp16 = weight.to(torch.float16)
    bias_fp16 = bias.to(torch.float16)
    
    # Padding 
    input_padded = F.pad(input_fp16, (padding, padding), "constant", 0)
    
    # FFT
    input_fft = rfft(input_padded, dim=-1)
    weight_fft = rfft(weight_fp16, dim=-1)
    
    # Convolution in frequency domain
    output_fft = input_fft * weight_fft
    
    # Inverse FFT
    output_padded = irfft(output_fft, dim=-1)
    
    # Strided extraction
    output_fp16 = output_padded[..., padding:-padding:stride].contiguous()
    
    # Bias addition and return to fp32
    output_fp16 = output_fp16 + bias_fp16.view(1, -1, 1, 1)
    return output_fp16.to(torch.float32)


function_signature = {
    "name": "torch_conv_fft_fp16",
    "inputs": [
        ((1, 1, 10, 10), torch.float32),
        ((1, 1, 3, 3), torch.float32),
        ((1, 1), torch.float32),
        (1, torch.int32),
        (1, torch.int32)
    ],
    "outputs": [
        ((1, 1, 10, 10), torch.float32),
    ]
}
