
import torch
import torch.fft

def robust_fourier_transform_fp16(input_tensor: torch.Tensor, loss_scale: float) -> torch.Tensor:
    """
    Performs a Fourier transform on the input tensor, applies a robust loss function, 
    and returns the result in FP16.
    """
    input_fp16 = input_tensor.to(torch.float16)
    fft_output = torch.fft.fft(input_fp16)
    loss = torch.abs(fft_output)
    loss = torch.where(loss > loss_scale, loss_scale, loss)  # Apply robust loss
    return loss.to(torch.float16)

function_signature = {
    "name": "robust_fourier_transform_fp16",
    "inputs": [
        ((128, 128), torch.float32),
        (torch.float32,)
    ],
    "outputs": [
        ((128, 128), torch.float16),
    ]
}
