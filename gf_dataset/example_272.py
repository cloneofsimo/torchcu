
import torch
import torch.nn.functional as F

def fused_gelu_example(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a fused operation of matrix multiplication, bias addition, and GELU activation.
    """
    output = torch.matmul(input_tensor, weight.t()) + bias
    return F.gelu(output)


function_signature = {
    "name": "fused_gelu_example",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
