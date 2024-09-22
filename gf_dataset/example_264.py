
import torch
import torch.nn.functional as F

def conv3d_fft_bf16_layer_scaling(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Performs a 3D convolution using FFT, with bfloat16 precision, reflection padding, and layer scaling.

    Args:
        input_tensor: Input tensor of shape (batch_size, in_channels, D, H, W).
        weight: Convolution kernel of shape (out_channels, in_channels, kernel_D, kernel_H, kernel_W).
        bias: Bias tensor of shape (out_channels).
        scale: Layer scaling factor.

    Returns:
        Output tensor of shape (batch_size, out_channels, D, H, W).
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    # Reflection padding
    input_bf16 = F.pad(input_bf16, (weight.shape[3] // 2, weight.shape[3] // 2,
                                    weight.shape[2] // 2, weight.shape[2] // 2,
                                    weight.shape[1] // 2, weight.shape[1] // 2), mode='reflect')

    # 3D convolution using FFT
    output_bf16 = torch.fft.irfft3(torch.fft.rfft3(input_bf16) * torch.fft.rfft3(weight_bf16, dim=(2, 3, 4)),
                                  dim=(2, 3, 4))

    # Layer scaling
    output_bf16 = output_bf16 * scale

    # Add bias
    output_bf16 = output_bf16 + bias_bf16.view(1, -1, 1, 1, 1)

    # ReLU activation
    output_bf16 = torch.relu(output_bf16)

    return output_bf16.to(torch.float32)

function_signature = {
    "name": "conv3d_fft_bf16_layer_scaling",
    "inputs": [
        ((2, 3, 10, 10, 10), torch.float32),
        ((4, 3, 3, 3, 3), torch.float32),
        ((4,), torch.float32),
        (torch.float32),
    ],
    "outputs": [
        ((2, 4, 10, 10, 10), torch.float32),
    ]
}
