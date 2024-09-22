
import torch
import torch.nn.functional as F
from torch.fft import irfft, rfft
from torch.nn.functional import max_pool1d

def torch_complex_topk_function(input_tensor: torch.Tensor, k: int) -> torch.Tensor:
    """
    Performs a complex-valued top-k operation after an inverse Fourier transform.
    """
    # Apply inverse FFT
    real_input = input_tensor.real.to(torch.bfloat16)
    imag_input = input_tensor.imag.to(torch.bfloat16)
    input_complex = torch.complex(real_input, imag_input)
    output_complex = irfft(input_complex, n=input_tensor.shape[-1], signal_ndim=1).to(torch.bfloat16)

    # Find top-k indices
    output_abs = output_complex.abs()
    _, indices = torch.topk(output_abs, k, dim=-1)

    # Extract top-k values
    output_complex = torch.gather(output_complex, dim=-1, index=indices).to(torch.float32)

    # Apply max pooling along the last dimension (time)
    output_complex = max_pool1d(output_complex.unsqueeze(1), kernel_size=k, stride=k).squeeze(1)

    return output_complex

function_signature = {
    "name": "torch_complex_topk_function",
    "inputs": [
        ((1, 128, 256), torch.complex64),
        (1, )
    ],
    "outputs": [
        ((1, 128, 64), torch.complex64),
    ]
}
