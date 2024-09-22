
import torch
import torch.fft

def torch_ifft_fp16_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs an inverse Fast Fourier Transform (IFFT) and returns the result in fp16.
    """
    input_fp16 = input_tensor.to(torch.float16)
    output = torch.fft.ifft(input_fp16, dim=-1)
    return output.to(torch.float16)

function_signature = {
    "name": "torch_ifft_fp16_function",
    "inputs": [
        ((8, 16), torch.complex64),
    ],
    "outputs": [
        ((8, 16), torch.float16),
    ]
}
