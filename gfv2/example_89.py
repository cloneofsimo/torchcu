
import torch

def rfft_int8_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Perform real-to-complex FFT and convert the output to int8.
    """
    output = torch.rfft(input_tensor, 1, normalized=True)
    return output.real.to(torch.int8)

function_signature = {
    "name": "rfft_int8_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.int8),
    ]
}
