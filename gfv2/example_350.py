
import torch

def prelu_bfloat16_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Perform a simple PReLU activation using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    output = torch.where(input_bf16 > 0, input_bf16, input_bf16 * weight_bf16)
    return output.to(torch.float32)

function_signature = {
    "name": "prelu_bfloat16_function",
    "inputs": [
        ((1,), torch.float32),
        ((1,), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
