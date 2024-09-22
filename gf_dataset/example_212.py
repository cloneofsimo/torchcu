
import torch

def torch_irfft_bfloat16_function(input_tensor: torch.Tensor, signal_ndim: int) -> torch.Tensor:
    """
    Perform an inverse real-to-complex fast Fourier transform (IRFFT) using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output = torch.irfft(input_bf16, signal_ndim=signal_ndim)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_irfft_bfloat16_function",
    "inputs": [
        ((4, 4, 2), torch.float32),
        (int, None)
    ],
    "outputs": [
        ((4, 4, 4), torch.float32)
    ]
}
