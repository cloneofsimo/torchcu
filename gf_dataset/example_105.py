
import torch
import torch.fft

def torch_image_laplacian_rrelu_irfft_bfloat16(image: torch.Tensor, kernel: torch.Tensor, rrelu_slope: float) -> torch.Tensor:
    """
    Applies a Laplacian filter to an image, applies RReLU activation, and then performs an inverse real-valued FFT.
    All operations are performed in bfloat16 precision for efficiency.
    """
    image_bf16 = image.to(torch.bfloat16)
    kernel_bf16 = kernel.to(torch.bfloat16)

    # Laplacian filtering
    filtered_image = torch.fft.irfft2(torch.fft.rfft2(image_bf16) * torch.fft.rfft2(kernel_bf16, image_bf16.shape))

    # RReLU activation
    filtered_image = torch.nn.functional.rrelu(filtered_image, lower=0.0, upper=rrelu_slope)

    # Inverse real-valued FFT
    output = torch.fft.irfft2(filtered_image)

    return output.to(torch.float32)

function_signature = {
    "name": "torch_image_laplacian_rrelu_irfft_bfloat16",
    "inputs": [
        ((1, 3, 256, 256), torch.float32),
        ((3, 3), torch.float32),
        (torch.float32,)
    ],
    "outputs": [
        ((1, 3, 256, 256), torch.float32),
    ]
}
