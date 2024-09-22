
import torch
import torch.nn.functional as F

def transposed_conv3d_envelope_fp16(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: list, padding: list) -> torch.Tensor:
    """
    Performs a transposed 3D convolution, applies signal envelope, and returns the result in FP16.

    Args:
        input_tensor: Input tensor of shape (batch_size, in_channels, D, H, W)
        weight: Weight tensor of shape (in_channels, out_channels, kernel_D, kernel_H, kernel_W)
        bias: Bias tensor of shape (out_channels)
        stride: Tuple of stride values for each dimension (stride_D, stride_H, stride_W)
        padding: Tuple of padding values for each dimension (padding_D, padding_H, padding_W)

    Returns:
        Output tensor of shape (batch_size, out_channels, D_out, H_out, W_out) in FP16.
    """

    # Convert to FP16
    input_tensor = input_tensor.to(torch.float16)
    weight = weight.to(torch.float16)
    bias = bias.to(torch.float16)

    # Transposed convolution
    output_tensor = F.conv_transpose3d(input_tensor, weight, bias, stride, padding)

    # Calculate signal envelope
    envelope = torch.abs(output_tensor)

    # Return envelope in FP16
    return envelope.to(torch.float16)

function_signature = {
    "name": "transposed_conv3d_envelope_fp16",
    "inputs": [
        ((2, 3, 10, 10, 10), torch.float32),
        ((3, 5, 3, 3, 3), torch.float32),
        ((5,), torch.float32),
        ((2, 2, 2), ),
        ((1, 1, 1), ),
    ],
    "outputs": [
        ((2, 5, 18, 18, 18), torch.float16),
    ]
}
