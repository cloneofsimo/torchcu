
import torch
import torch.fft

def conv_fft_3d_normal_function(input_tensor: torch.Tensor, kernel: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Performs a 3D convolution using FFT with normalization.
    """
    # Normalize input
    input_tensor = (input_tensor - mean) / std
    
    # Pad input to match kernel size (assuming kernel is smaller)
    pad_size = (kernel.shape[2] // 2, kernel.shape[2] // 2,
                kernel.shape[1] // 2, kernel.shape[1] // 2,
                kernel.shape[0] // 2, kernel.shape[0] // 2)
    input_tensor = torch.nn.functional.pad(input_tensor, pad_size, 'constant')

    # Perform FFT convolution
    input_fft = torch.fft.fftn(input_tensor, dim=(2, 3, 4))
    kernel_fft = torch.fft.fftn(kernel, dim=(2, 3, 4))
    output_fft = input_fft * kernel_fft
    output = torch.fft.ifftn(output_fft, dim=(2, 3, 4)).real

    # Crop output to original size
    output = output[:, :, pad_size[0]:-pad_size[1], pad_size[2]:-pad_size[3], pad_size[4]:-pad_size[5]]

    return output.view(input_tensor.shape[0], -1)

function_signature = {
    "name": "conv_fft_3d_normal_function",
    "inputs": [
        ((16, 64, 16, 16, 16), torch.float32),
        ((3, 3, 3), torch.float32),
        (torch.float32),
        (torch.float32),
    ],
    "outputs": [
        ((16, 3072), torch.float32),
    ]
}
