
import torch
import torch.fft

def complex_mse_loss_with_idft(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate the mean squared error (MSE) between the inverse discrete Fourier transform (IDFT) of two complex tensors.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    target_bf16 = target_tensor.to(torch.bfloat16)

    input_idft = torch.fft.irfft(input_bf16, dim=-1).to(torch.float32)
    target_idft = torch.fft.irfft(target_bf16, dim=-1).to(torch.float32)

    mse_loss = torch.mean(torch.square(input_idft - target_idft))

    return mse_loss

function_signature = {
    "name": "complex_mse_loss_with_idft",
    "inputs": [
        ((10, 10, 128), torch.complex64),
        ((10, 10, 128), torch.complex64)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
