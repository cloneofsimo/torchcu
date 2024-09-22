
import torch
import torch.nn.functional as F
from torch.fft import irfft2

def torch_separable_conv_and_fourier_function(input_tensor: torch.Tensor, weight_depthwise: torch.Tensor, weight_pointwise: torch.Tensor) -> torch.Tensor:
    """
    Applies a separable convolution followed by an inverse 2D Fourier transform.
    """
    # Separable convolution
    output = F.conv2d(input_tensor, weight_depthwise, groups=input_tensor.shape[1], padding='same')
    output = F.conv2d(output, weight_pointwise, padding='same')
    
    # Inverse 2D Fourier transform
    output = irfft2(output, dim=(-2, -1))
    
    # Apply ReLU activation inplace
    output.relu_() 
    
    return output

function_signature = {
    "name": "torch_separable_conv_and_fourier_function",
    "inputs": [
        ((1, 3, 32, 32), torch.float32), 
        ((3, 1, 3, 3), torch.float32),
        ((16, 3, 1, 1), torch.float32)
    ],
    "outputs": [
        ((1, 16, 32, 32), torch.float32),
    ]
}
