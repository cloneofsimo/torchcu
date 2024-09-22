
import torch

def torch_where_transpose_bf16_function(input_tensor: torch.Tensor, condition: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """
    Performs a where operation with bfloat16 precision, then transposes the result. 
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    condition_bf16 = condition.to(torch.bfloat16)
    other_bf16 = other.to(torch.bfloat16)
    output_bf16 = torch.where(condition_bf16, input_bf16, other_bf16)
    output = output_bf16.t().to(torch.float32)
    return output

function_signature = {
    "name": "torch_where_transpose_bf16_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.bool),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
