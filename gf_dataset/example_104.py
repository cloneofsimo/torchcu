
import torch
import torch.fft

def torch_ifft_bfloat16_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse fast Fourier transform (IFFT) using bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output = torch.fft.ifft(input_bf16, dim=-1)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_ifft_bfloat16_function",
    "inputs": [
        ((4, 4), torch.complex64)
    ],
    "outputs": [
        ((4, 4), torch.complex64),
    ]
}
