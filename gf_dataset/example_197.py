
import torch

def torch_rfft_bfloat16_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a real-to-complex FFT using bfloat16 for improved performance.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output = torch.fft.rfft(input_bf16, dim=-1)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_rfft_bfloat16_function",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 2), torch.float32),
    ]
}
