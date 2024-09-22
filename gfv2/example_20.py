
import torch
import torch.nn.functional as F

def conv3d_fft_replication_pad(input_tensor: torch.Tensor, weight: torch.Tensor, 
                                  padding: int, stride: int = 1, dilation: int = 1) -> torch.Tensor:
    """
    Applies a 3D convolution using FFT and replication padding.
    """
    # Replication padding
    input_tensor = F.pad(input_tensor, (padding, padding, padding, padding, padding, padding), mode='replicate')

    # 3D FFT
    input_tensor = torch.fft.fft3(input_tensor)
    weight_fft = torch.fft.fft3(weight)

    # Convolution in frequency domain
    output_fft = torch.fft.ifft3(input_tensor * weight_fft)

    # Crop output to original size
    output_tensor = output_fft[padding:padding + input_tensor.shape[2], 
                                padding:padding + input_tensor.shape[3], 
                                padding:padding + input_tensor.shape[4]]

    # ReLU activation
    return F.relu(output_tensor)

function_signature = {
    "name": "conv3d_fft_replication_pad",
    "inputs": [
        ((4, 4, 4, 4, 4), torch.float32),  # input tensor (batch, channels, D, H, W)
        ((4, 4, 3, 3, 3), torch.float32)   # weight tensor (out_channels, in_channels, kernel_D, kernel_H, kernel_W)
    ],
    "outputs": [
        ((4, 4, 4, 4, 4), torch.float32)  # output tensor
    ]
}
