
import torch
import torch.nn.functional as F

def dynamic_conv_rrelu_bf16(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, bucket_sizes: list, lower: float, upper: float) -> torch.Tensor:
    """
    Performs a dynamic convolution, ReLU, and bucketing using bfloat16 precision.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (B, C_in, H, W).
        weight (torch.Tensor): Weight tensor of shape (C_out, C_in, K, K).
        bias (torch.Tensor): Bias tensor of shape (C_out).
        bucket_sizes (list): List of bucket sizes for bucketing.
        lower (float): Lower bound for RReLU activation.
        upper (float): Upper bound for RReLU activation.

    Returns:
        torch.Tensor: Output tensor of shape (B, C_out, H, W).
    """

    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    output = F.conv2d(input_bf16, weight_bf16, bias_bf16, padding=1)
    output = F.rrelu(output, lower, upper)
    output = torch.bucketize(output.float(), torch.tensor(bucket_sizes)).to(torch.bfloat16)

    return output.to(torch.float32)

function_signature = {
    "name": "dynamic_conv_rrelu_bf16",
    "inputs": [
        ((1, 3, 16, 16), torch.float32),
        ((16, 3, 3, 3), torch.float32),
        ((16,), torch.float32),
        ([1, 2, 3, 5, 10, 15, 20, 30], torch.int32),
        (0.1, torch.float32),
        (0.3, torch.float32)
    ],
    "outputs": [
        ((1, 16, 16, 16), torch.float32),
    ]
}
