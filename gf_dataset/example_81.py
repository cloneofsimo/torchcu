
import torch

def bilateral_filter_fp32(image: torch.Tensor, kernel_size: int, sigma_spatial: float, sigma_color: float) -> torch.Tensor:
    """
    Applies a bilateral filter to the input image.

    Args:
        image (torch.Tensor): The input image with shape (B, C, H, W).
        kernel_size (int): The size of the kernel (must be odd).
        sigma_spatial (float): The standard deviation for the spatial Gaussian kernel.
        sigma_color (float): The standard deviation for the color Gaussian kernel.

    Returns:
        torch.Tensor: The filtered image with the same shape as the input image.
    """

    # Calculate Gaussian kernel weights for spatial and color dimensions
    spatial_kernel = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
    spatial_kernel = torch.exp(-(spatial_kernel**2) / (2 * sigma_spatial**2))
    spatial_kernel = spatial_kernel / spatial_kernel.sum()  # Normalize

    color_kernel = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
    color_kernel = torch.exp(-(color_kernel**2) / (2 * sigma_color**2))
    color_kernel = color_kernel / color_kernel.sum()  # Normalize

    # Perform the bilateral filtering
    filtered_image = torch.zeros_like(image)
    for b in range(image.shape[0]):
        for c in range(image.shape[1]):
            for h in range(image.shape[2]):
                for w in range(image.shape[3]):
                    # Calculate weights for each pixel in the neighborhood
                    weights = spatial_kernel[None, None, :, :] * color_kernel[None, None, None, :]
                    for kh in range(-(kernel_size // 2), (kernel_size // 2) + 1):
                        for kw in range(-(kernel_size // 2), (kernel_size // 2) + 1):
                            # Calculate color distance
                            color_dist = (image[b, c, h + kh, w + kw] - image[b, c, h, w]).abs()
                            # Multiply with color kernel
                            weights[0, 0, kh + (kernel_size // 2), kw + (kernel_size // 2)] *= torch.exp(-(color_dist**2) / (2 * sigma_color**2))
                    # Normalize weights
                    weights = weights / weights.sum()

                    # Calculate filtered pixel value
                    filtered_image[b, c, h, w] = (image[b, c, h + kh, w + kw] * weights[0, 0, kh + (kernel_size // 2), kw + (kernel_size // 2)]).sum()

    return filtered_image

function_signature = {
    "name": "bilateral_filter_fp32",
    "inputs": [
        ((1, 3, 128, 128), torch.float32),
        (1, torch.int32),
        (1, torch.float32),
        (1, torch.float32)
    ],
    "outputs": [
        ((1, 3, 128, 128), torch.float32)
    ]
}
