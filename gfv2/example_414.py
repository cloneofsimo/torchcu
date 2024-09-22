
import torch

def my_complex_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a complex operation involving matrix multiplication, 
    ReLU activation, and zero padding. 
    """
    output = torch.matmul(input_tensor, weight.t())
    output = torch.relu(output)
    output = torch.nn.functional.pad(output, (2, 2, 2, 2), "constant", 0)
    return output.to(torch.bfloat16)

function_signature = {
    "name": "my_complex_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((8, 8), torch.bfloat16),
    ]
}
