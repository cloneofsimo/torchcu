
import torch

def torch_elementwise_div_function(input_tensor: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
    """
    Performs element-wise division with broadcasting support using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    divisor_bf16 = divisor.to(torch.bfloat16)
    output = torch.true_divide(input_bf16, divisor_bf16)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_elementwise_div_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
