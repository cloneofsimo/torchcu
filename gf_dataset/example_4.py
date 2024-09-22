
import torch
import torch.fft

def torch_fft_conv3d_function(input_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Perform 3D convolution using FFT with fp16 precision.
    """
    input_tensor_fp16 = input_tensor.to(torch.float16)
    kernel_fp16 = kernel.to(torch.float16)
    
    # Calculate FFT of input and kernel
    input_fft = torch.fft.fftn(input_tensor_fp16, dim=[-3, -2, -1])
    kernel_fft = torch.fft.fftn(kernel_fp16, dim=[-3, -2, -1])
    
    # Perform element-wise multiplication in frequency domain
    output_fft = input_fft * kernel_fft
    
    # Inverse FFT to get convolution result
    output_tensor_fp16 = torch.fft.ifftn(output_fft, dim=[-3, -2, -1])
    
    # Convert back to float32
    output_tensor = output_tensor_fp16.to(torch.float32)
    
    return output_tensor

function_signature = {
    "name": "torch_fft_conv3d_function",
    "inputs": [
        ((2, 3, 10, 10, 10), torch.float32),
        ((3, 3, 3), torch.float32)
    ],
    "outputs": [
        ((2, 3, 10, 10, 10), torch.float32),
    ]
}
