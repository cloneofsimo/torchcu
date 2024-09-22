
import torch
import torch.fft

def my_complex_function(input_tensor: torch.Tensor, kernel: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a 3D convolution using FFT, applies a bias, clamps the result,
    and returns the output tensor.

    Args:
        input_tensor: The input tensor of shape (B, C, D, H, W).
        kernel: The convolution kernel of shape (C, C, K_D, K_H, K_W).
        bias: The bias tensor of shape (C).

    Returns:
        The output tensor of shape (B, C, D, H, W) after convolution, bias addition,
        and clamping.
    """
    # Pad the input tensor to match the kernel size
    padding = (kernel.shape[2] // 2, kernel.shape[3] // 2, kernel.shape[4] // 2)
    input_tensor = torch.nn.functional.pad(input_tensor, padding, mode="constant")

    # Perform 3D convolution using FFT
    input_fft = torch.fft.fft(input_tensor, dim=(2, 3, 4))
    kernel_fft = torch.fft.fft(kernel, dim=(2, 3, 4))
    output_fft = input_fft * kernel_fft
    output_tensor = torch.fft.ifft(output_fft, dim=(2, 3, 4))

    # Apply bias
    output_tensor = output_tensor + bias.view(1, -1, 1, 1, 1)

    # Clamp the result to the range [0, 1]
    output_tensor.clamp_(0, 1)

    # Return the output tensor
    return output_tensor.float()

function_signature = {
    "name": "my_complex_function",
    "inputs": [
        ((2, 4, 8, 8, 8), torch.float32),
        ((4, 4, 3, 3, 3), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((2, 4, 8, 8, 8), torch.float32),
    ]
}
