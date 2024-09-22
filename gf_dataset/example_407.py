
import torch

def torch_softshrink_fftshift_function(input_tensor: torch.Tensor, lambd: float) -> torch.Tensor:
    """
    Applies the soft shrinkage function and then performs a FFT shift.
    """
    output = torch.nn.functional.softshrink(input_tensor, lambd)
    output = torch.fft.fftshift(output, dim=(-1, -2))  
    return output

function_signature = {
    "name": "torch_softshrink_fftshift_function",
    "inputs": [
        ((4, 4), torch.float32),
        (float, torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
