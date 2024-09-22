
import torch
import torch.fft

def torch_image_laplacian_fft(image: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Laplacian of an image using FFT.

    Args:
        image: A 2D or 3D tensor representing the image (H x W or C x H x W).

    Returns:
        A tensor of the same shape as the input image, containing the Laplacian.
    """
    # Move to the device and convert to complex
    image = image.to(torch.complex64)
    
    # Calculate the 2D FFT
    fft_image = torch.fft.fft2(image)
    
    # Create frequency domain kernel
    height, width = image.shape[-2:]
    x = torch.arange(width)
    y = torch.arange(height)
    fx = torch.fft.fftfreq(width, d=1.0/width)
    fy = torch.fft.fftfreq(height, d=1.0/height)
    fx, fy = torch.meshgrid(fx, fy)
    laplacian_kernel = -4 * torch.pi**2 * (fx**2 + fy**2)
    
    # Apply Laplacian kernel
    laplacian_image = laplacian_kernel * fft_image
    
    # Inverse FFT
    laplacian_image = torch.fft.ifft2(laplacian_image)
    
    # Return real component
    return laplacian_image.real.to(torch.float32)

function_signature = {
    "name": "torch_image_laplacian_fft",
    "inputs": [
        ((3, 256, 256), torch.float32)
    ],
    "outputs": [
        ((3, 256, 256), torch.float32)
    ]
}
