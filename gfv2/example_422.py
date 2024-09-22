
import torch

def my_function(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor, input_tensor3: torch.Tensor) -> torch.Tensor:
    """
    Performs a batched matrix multiplication followed by element-wise addition with broadcasting.
    """
    output = torch.einsum('ijk,kl->ijl', input_tensor1, input_tensor2)
    output = output + input_tensor3
    return output

function_signature = {
    "name": "my_function",
    "inputs": [
        ((2, 3, 4), torch.float32),
        ((4, 5), torch.float32),
        ((2, 3, 5), torch.float32)
    ],
    "outputs": [
        ((2, 3, 5), torch.float32)
    ]
}
