
import torch

def torch_fft_bfloat16_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Perform a fast fourier transform on the input tensor using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output = torch.fft.fft(input_bf16)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_fft_bfloat16_function",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.complex64),
    ]
}
