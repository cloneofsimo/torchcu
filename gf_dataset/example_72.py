
import torch
from torch.fft import fft, ifft
from torch.cuda.amp import autocast

def torch_complex_dft_gt_ceil_function(input_tensor: torch.Tensor, gt: float) -> torch.Tensor:
    """
    Performs a complex DFT, applies a greater-than comparison with a threshold, 
    and then uses the ceil function to round the result up to the nearest integer. 
    All operations are performed in fp16.
    """
    with autocast():
        input_fp16 = input_tensor.to(torch.float16)
        output = fft(input_fp16)
        output = torch.gt(torch.abs(output), gt).to(torch.float16)
        output = torch.ceil(output)
        output = ifft(output)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_complex_dft_gt_ceil_function",
    "inputs": [
        ((16, 16), torch.complex64), 
    ],
    "outputs": [
        ((16, 16), torch.complex64),
    ]
}
