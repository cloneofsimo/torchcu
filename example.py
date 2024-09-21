import torch

def torch_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Perform a simple linear transformation (matrix multiplication) and activation using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    output = torch.matmul(input_bf16, weight_bf16.t())
    return torch.relu(output).to(torch.float32)

function_signature = [
    'torch_function',
    ('input_tensor', (4, 4), torch.float32),
    ('weight', (4, 4), torch.float32)
]