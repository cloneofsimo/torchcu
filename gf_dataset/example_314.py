
import torch
import torch.nn.functional as F

def bilateral_filter_int8(input_tensor: torch.Tensor, kernel_size: int, sigma_color: float, sigma_spatial: float) -> torch.Tensor:
    """
    Applies a bilateral filter to an image.

    Args:
        input_tensor: The input image tensor.
        kernel_size: The size of the kernel (must be odd).
        sigma_color: The standard deviation of the color kernel.
        sigma_spatial: The standard deviation of the spatial kernel.

    Returns:
        The filtered image tensor.
    """
    # Convert to int8
    input_tensor_int8 = input_tensor.to(torch.int8)
    # Apply bilateral filter with int8 precision
    output_int8 = F.bilateral_filter(input_tensor_int8, kernel_size, sigma_color, sigma_spatial)
    # Convert back to float
    output = output_int8.to(torch.float32)
    return output

function_signature = {
    "name": "bilateral_filter_int8",
    "inputs": [
        ((3, 224, 224), torch.float32),
        (1, torch.int32),
        (1, torch.float32),
        (1, torch.float32)
    ],
    "outputs": [
        ((3, 224, 224), torch.float32),
    ]
}
