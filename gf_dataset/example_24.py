
import torch
import torch.nn as nn

def torch_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on the input tensor:
        1. Adaptive average pooling in 2D
        2. Dot product with a weight tensor
        3. Addition of bias
        4. Inverse real-to-complex FFT (irfft)
        5. Transposed convolution in 3D
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    # Adaptive average pooling
    x = nn.functional.adaptive_avg_pool2d(input_bf16, (4, 4))

    # Dot product
    x = torch.dot(x, weight_bf16)

    # Bias addition
    x = x + bias_bf16

    # Inverse real-to-complex FFT
    x = torch.irfft(x, 2, signal_ndim=1)

    # Transposed convolution
    x = nn.functional.conv_transpose3d(x.unsqueeze(1), weight_bf16.unsqueeze(0), bias_bf16.unsqueeze(0))

    return x.to(torch.float32)

function_signature = {
    "name": "torch_function",
    "inputs": [
        ((8, 16, 32, 32), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32),
    ],
    "outputs": [
        ((8, 1, 4, 4, 4), torch.float32),
    ]
}
