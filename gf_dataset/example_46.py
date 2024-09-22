
import torch

def torch_fft_conv1d_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs 1D convolution using FFT for efficient computation.
    """
    input_tensor = input_tensor.to(torch.complex64)
    weight = weight.to(torch.complex64)
    
    # Pad the input signal for valid convolution
    input_tensor = torch.nn.functional.pad(input_tensor, (weight.shape[2] - 1, 0), mode='constant')
    
    # Perform FFT on both input and weight
    input_fft = torch.fft.fft(input_tensor, dim=1)
    weight_fft = torch.fft.fft(weight, dim=2)
    
    # Perform element-wise multiplication in frequency domain
    output_fft = input_fft * weight_fft.unsqueeze(1)
    
    # Perform inverse FFT to get the convolution result
    output = torch.fft.ifft(output_fft, dim=1)
    
    # Return the real part of the output
    return output.real.to(torch.float32)

function_signature = {
    "name": "torch_fft_conv1d_function",
    "inputs": [
        ((1, 10, 4), torch.float32),
        ((1, 1, 3, 4), torch.float32)
    ],
    "outputs": [
        ((1, 10, 1), torch.float32),
    ]
}
