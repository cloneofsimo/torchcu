
import torch

def torch_sqrt_einsum_transpose(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Calculates the square root of the input tensor, performs an einsum operation
    with the weight tensor (transposed), and returns the result.
    """
    input_tensor.sqrt_()  # In-place square root
    output = torch.einsum('ij,jk->ik', input_tensor, weight.t())
    return output

function_signature = {
    "name": "torch_sqrt_einsum_transpose",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
