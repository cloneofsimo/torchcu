
import torch

def torch_einsum_outer_bfloat16_function(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the outer product using einsum and bfloat16.
    """
    input1_bf16 = input_tensor1.to(torch.bfloat16)
    input2_bf16 = input_tensor2.to(torch.bfloat16)
    output = torch.einsum('i,j->ij', input1_bf16, input2_bf16)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_einsum_outer_bfloat16_function",
    "inputs": [
        ((4,), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
