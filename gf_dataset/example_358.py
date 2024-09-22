
import torch
import torch.fft

def torch_dft_bfloat16_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a 1D DFT on a batch of signals using bfloat16 precision. 
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output_bf16 = torch.fft.fft(input_bf16, dim=1)
    return output_bf16.to(torch.float32)

function_signature = {
    "name": "torch_dft_bfloat16_function",
    "inputs": [
        ((16, 32), torch.float32)
    ],
    "outputs": [
        ((16, 32), torch.float32),
    ]
}
