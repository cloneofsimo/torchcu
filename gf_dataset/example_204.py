
import torch

def fused_linear_bf16(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a fused linear operation with bfloat16 precision and returns the result in float32.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)
    output_bf16 = torch.nn.functional.linear(input_bf16, weight_bf16, bias_bf16)
    return output_bf16.to(torch.float32)

function_signature = {
    "name": "fused_linear_bf16",
    "inputs": [
        ((10, 128), torch.float32),
        ((128, 512), torch.float32),
        ((512,), torch.float32)
    ],
    "outputs": [
        ((10, 512), torch.float32),
    ]
}
