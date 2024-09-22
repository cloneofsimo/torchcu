
import torch

def torch_elementwise_add_bf16_function(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """
    Performs element-wise addition using bfloat16.
    """
    input1_bf16 = input1.to(torch.bfloat16)
    input2_bf16 = input2.to(torch.bfloat16)
    output = input1_bf16 + input2_bf16
    return output.to(torch.float32)

function_signature = {
    "name": "torch_elementwise_add_bf16_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
