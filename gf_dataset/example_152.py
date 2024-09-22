
import torch
import torch.fft

def torch_fft_conv2d(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Performs a 2D convolution using FFTs for efficient computation.
    """
    # Pad the input to match the kernel size
    input_padded = torch.nn.functional.pad(input, (kernel.shape[2]//2, kernel.shape[2]//2, kernel.shape[1]//2, kernel.shape[1]//2), 'constant', 0)
    
    # Compute FFTs
    input_fft = torch.fft.fft2(input_padded)
    kernel_fft = torch.fft.fft2(kernel, s=input_padded.shape[2:])
    
    # Multiply FFTs
    output_fft = input_fft * kernel_fft
    
    # Inverse FFT
    output = torch.fft.ifft2(output_fft)
    
    # Crop to original size
    output = output[..., kernel.shape[2]//2:output.shape[2]-kernel.shape[2]//2, kernel.shape[1]//2:output.shape[1]-kernel.shape[1]//2]
    
    # Return real part
    return output.real

function_signature = {
    "name": "torch_fft_conv2d",
    "inputs": [
        ((1, 3, 128, 128), torch.float32),
        ((1, 3, 3, 3), torch.float32)
    ],
    "outputs": [
        ((1, 3, 128, 128), torch.float32)
    ]
}
