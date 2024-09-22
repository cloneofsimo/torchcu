
import torch
import torch.nn.functional as F
from torch.fft import fftn, ifftn

def torch_3d_processing_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of 3D operations on the input tensor:
    1. Max Pooling 3D
    2. DFT (Discrete Fourier Transform)
    3. Transposed Convolution 3D
    4. Norm calculation
    5. ReLU Activation
    """
    # Convert to bfloat16 for faster computations
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)

    # Max Pooling 3D
    pooled = F.max_pool3d(input_bf16, kernel_size=3, stride=2, padding=1)

    # DFT (Discrete Fourier Transform)
    dft_output = fftn(pooled, dim=(-3, -2, -1))

    # Transposed Convolution 3D
    transposed_output = F.conv_transpose3d(dft_output, weight_bf16, stride=2, padding=1, output_padding=1)

    # Norm calculation
    norm_output = torch.norm(transposed_output, dim=(-3, -2, -1))

    # ReLU activation
    output = torch.relu(norm_output).to(torch.float32)

    return output

function_signature = {
    "name": "torch_3d_processing_function",
    "inputs": [
        ((16, 16, 16, 3), torch.float32),
        ((3, 3, 3, 3), torch.float32)
    ],
    "outputs": [
        ((16, 16, 16), torch.float32),
    ]
}
