
import torch

def fft_conv3d_inplace(input_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Performs a 3D convolution using FFT. Modifies the input tensor inplace.
    """
    # Ensure kernel is padded to match input size
    kernel_size = kernel.shape
    input_size = input_tensor.shape
    pad_sizes = [(input_size[i] - kernel_size[i]) // 2 for i in range(3)]
    kernel = torch.nn.functional.pad(kernel, [pad_sizes[2], pad_sizes[2], pad_sizes[1], pad_sizes[1], pad_sizes[0], pad_sizes[0]])

    # Perform 3D FFT on both input and kernel
    input_fft = torch.fft.fftn(input_tensor, dim=(1, 2, 3))
    kernel_fft = torch.fft.fftn(kernel, dim=(0, 1, 2))

    # Element-wise multiplication in frequency domain
    output_fft = input_fft * kernel_fft

    # Perform inverse FFT
    output_tensor = torch.fft.ifftn(output_fft, dim=(1, 2, 3)).real

    # Copy result back to input tensor (inplace)
    input_tensor[:] = output_tensor
    return input_tensor

function_signature = {
    "name": "fft_conv3d_inplace",
    "inputs": [
        ((2, 3, 4, 5), torch.float32),
        ((3, 2, 2, 2), torch.float32)
    ],
    "outputs": [
        ((2, 3, 4, 5), torch.float32)
    ]
}

