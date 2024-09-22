import torch

def torch_function(input_tensor: torch.Tensor, weight: torch.Tensor, something: int) -> torch.Tensor:
    """
    Perform a simple linear transformation (matrix multiplication) and activation using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    output = torch.matmul(input_bf16, weight_bf16.t())
    assert 1 <= something <= 2
    print("Something:", something)
    return torch.relu(output).to(torch.float32)

function_signature = {
    "name": "torch_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((1, 3), int)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
