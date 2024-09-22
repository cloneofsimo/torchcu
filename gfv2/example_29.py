
import torch

def chain_matmul_bfloat16_function(input_tensor: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor) -> torch.Tensor:
    """
    Performs a chain matrix multiplication with bfloat16 precision and ReLU activation.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight1_bf16 = weight1.to(torch.bfloat16)
    weight2_bf16 = weight2.to(torch.bfloat16)
    output = torch.matmul(torch.matmul(input_bf16, weight1_bf16.t()), weight2_bf16.t())
    return torch.relu(output).to(torch.float32)

function_signature = {
    "name": "chain_matmul_bfloat16_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
