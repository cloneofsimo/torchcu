
import torch

def torch_einsum_inner_function(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor, input_tensor3: torch.Tensor) -> torch.Tensor:
    """
    Perform an einsum operation with inner product.
    """
    return torch.einsum('ijk,klm,mnp->ijnp', input_tensor1, input_tensor2, input_tensor3)

function_signature = {
    "name": "torch_einsum_inner_function",
    "inputs": [
        ((2, 3, 4), torch.float32),
        ((4, 5, 6), torch.float32),
        ((6, 7, 8), torch.float32)
    ],
    "outputs": [
        ((2, 3, 5, 8), torch.float32),
    ]
}
