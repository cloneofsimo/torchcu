
import torch
import torch.nn.functional as F

def lightweight_conv_ssl_min_index_select_bf16(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Lightweight convolution for self-supervised learning, applying min pooling and index selection.

    Args:
        input_tensor: Input tensor of shape (B, C_in, H, W).
        weight: Weight tensor of shape (C_out, C_in, kernel_size, kernel_size).
        bias: Bias tensor of shape (C_out).
        indices: Indices tensor of shape (B, H_out, W_out).

    Returns:
        Output tensor of shape (B, C_out, H_out, W_out).
    """
    # Convert to bfloat16
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    # Perform lightweight convolution using CUDNN
    output_bf16 = F.conv2d(input_bf16, weight_bf16, bias_bf16, padding=1)

    # Apply minimum pooling
    output_bf16 = torch.min(output_bf16, dim=1, keepdim=True).values

    # Index select based on provided indices
    output_bf16 = torch.gather(output_bf16, dim=1, index=indices.unsqueeze(1).long())

    # Convert back to float32
    return output_bf16.to(torch.float32)

function_signature = {
    "name": "lightweight_conv_ssl_min_index_select_bf16",
    "inputs": [
        ((16, 32, 224, 224), torch.float32),
        ((32, 32, 3, 3), torch.float32),
        ((32,), torch.float32),
        ((16, 56, 56), torch.int64)
    ],
    "outputs": [
        ((16, 32, 56, 56), torch.float32)
    ]
}
