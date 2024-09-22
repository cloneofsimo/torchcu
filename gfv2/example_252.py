
import torch
import torch.fft

def conv1d_instance_norm_sigmoid_bf16(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a 1D convolution with instance normalization, sigmoid activation, and bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    # Convolution with FFT
    output = torch.fft.irfft(torch.fft.rfft(input_bf16, dim=1) * torch.fft.rfft(weight_bf16, dim=1), dim=1)

    # Instance normalization
    mean = output.mean(dim=1, keepdim=True)
    std = output.std(dim=1, keepdim=True)
    output = (output - mean) / (std + 1e-5)  # Add small constant for numerical stability

    # Bias and activation
    output = torch.sigmoid(output + bias_bf16).to(torch.float32)

    return output

function_signature = {
    "name": "conv1d_instance_norm_sigmoid_bf16",
    "inputs": [
        ((16, 1024), torch.float32),
        ((1024, 1024), torch.float32),
        ((1024,), torch.float32)
    ],
    "outputs": [
        ((16, 1024), torch.float32),
    ]
}
