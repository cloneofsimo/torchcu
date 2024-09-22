
import torch

def complex_max_irfft_function(input_tensor: torch.Tensor, signal_length: int) -> torch.Tensor:
    """
    Performs a complex max operation followed by an inverse real-to-complex FFT (iRFFT) on the input tensor.
    """
    complex_tensor = input_tensor.complex()
    max_values = torch.max(complex_tensor.real, complex_tensor.imag)
    
    # Use inplace iRFFT for efficiency
    output = torch.irfft(max_values, signal_length, signal_ndim=1, normalized=True, onesided=True)
    return output

function_signature = {
    "name": "complex_max_irfft_function",
    "inputs": [
        ((1, 16, 128), torch.complex64),
        (128, )
    ],
    "outputs": [
        ((1, 16, 128), torch.float32),
    ]
}
