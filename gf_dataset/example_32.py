
import torch

def torch_fft_fp16_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a Fast Fourier Transform (FFT) on the input tensor and returns the result in fp16.
    """
    return torch.fft.fft(input_tensor).to(torch.float16)

function_signature = {
    "name": "torch_fft_fp16_function",
    "inputs": [
        ((16, 16), torch.float32),
    ],
    "outputs": [
        ((16, 16), torch.float16),
    ]
}
