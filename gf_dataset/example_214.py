
import torch

def torch_einsum_transpose_fp32_function(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor) -> torch.Tensor:
    """
    Performs an einsum operation with transposition on two input tensors, returning a float32 tensor.
    """
    return torch.einsum('ijk,kl->ijl', input_tensor1, input_tensor2.transpose(1, 2)).float()

function_signature = {
    "name": "torch_einsum_transpose_fp32_function",
    "inputs": [
        ((4, 3, 2), torch.float32),
        ((2, 5), torch.float32)
    ],
    "outputs": [
        ((4, 3, 5), torch.float32),
    ]
}
