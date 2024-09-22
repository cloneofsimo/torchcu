
import torch
import torch.fft

def conv_fft_bfloat16_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs convolution using FFT with bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    
    # FFT Padding
    input_padded = torch.fft.fft(torch.nn.functional.pad(input_bf16, (weight.shape[2] // 2, weight.shape[2] // 2)), dim=2)
    weight_padded = torch.fft.fft(weight_bf16, dim=2)

    # Frequency-domain multiplication
    output_fft = input_padded * weight_padded.unsqueeze(0)

    # Inverse FFT
    output = torch.fft.ifft(output_fft, dim=2).real.to(torch.float32)
    
    return output

function_signature = {
    "name": "conv_fft_bfloat16_function",
    "inputs": [
        ((1, 4, 8), torch.float32),  # (batch, channels, input_size)
        ((1, 4, 3), torch.float32)   # (out_channels, in_channels, kernel_size)
    ],
    "outputs": [
        ((1, 4, 10), torch.float32) # (batch, out_channels, output_size)
    ]
}
