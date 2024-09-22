
import torch

def torch_complex_idft_norm_flatten_int8_fp16(input_tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Performs a complex inverse discrete Fourier transform (IDFT), normalizes the output, flattens it, 
    quantizes to int8, and returns the result in fp16.

    Args:
        input_tensor (torch.Tensor): Complex input tensor.
        scale (float): Scaling factor for normalization.

    Returns:
        torch.Tensor: Quantized and normalized output tensor in fp16.
    """
    output = torch.fft.ifft(input_tensor, dim=-1)  # IDFT
    output = output.real  # Take the real part
    output.div_(scale)  # Normalize
    output = output.flatten()  # Flatten
    output = output.to(torch.int8)  # Quantize to int8
    return output.to(torch.float16)  # Convert to fp16

function_signature = {
    "name": "torch_complex_idft_norm_flatten_int8_fp16",
    "inputs": [
        ((4, 4, 2), torch.complex64),
        ((), torch.float32)
    ],
    "outputs": [
        ((32,), torch.float16),
    ]
}
