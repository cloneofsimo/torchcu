import torch

def torch_linear_bfloat16_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Perform a simple linear transformation (matrix multiplication) and activation using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    output = torch.matmul(input_bf16, weight_bf16.t())
    return torch.relu(output).to(torch.float32)

function_signature = {
    "name": "torch_linear_bfloat16_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}