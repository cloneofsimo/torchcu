
import torch

def torch_clamp_bmm_bf16_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                                  min_value: float, max_value: float) -> torch.Tensor:
    """
    Performs a batched matrix multiplication, clamps the result, and applies bias using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    output_bf16 = torch.bmm(input_bf16, weight_bf16)
    output_bf16.clamp_(min_value, max_value)
    output_bf16.add_(bias_bf16)

    return output_bf16.to(torch.float32)

function_signature = {
    "name": "torch_clamp_bmm_bf16_function",
    "inputs": [
        ((2, 3, 4), torch.float32),
        ((4, 5), torch.float32),
        ((5,), torch.float32),
        (0.0, torch.float32),
        (1.0, torch.float32)
    ],
    "outputs": [
        ((2, 3, 5), torch.float32)
    ]
}
