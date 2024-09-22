
import torch

def chain_matmul_function(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor, input_tensor3: torch.Tensor) -> torch.Tensor:
    """
    Performs a chain matrix multiplication of three input tensors using bfloat16 precision.
    """
    input_bf16_1 = input_tensor1.to(torch.bfloat16)
    input_bf16_2 = input_tensor2.to(torch.bfloat16)
    input_bf16_3 = input_tensor3.to(torch.bfloat16)
    output = torch.matmul(input_bf16_1, input_bf16_2)
    output = torch.matmul(output, input_bf16_3)
    return output.to(torch.float32)

function_signature = {
    "name": "chain_matmul_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
