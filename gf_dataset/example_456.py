
import torch

def torch_frobenius_norm_bf16(input_tensor: torch.Tensor, dim: int = None) -> torch.Tensor:
    """
    Calculates the Frobenius norm of a tensor in bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    norm = torch.linalg.norm(input_bf16, ord="fro", dim=dim)
    return norm.to(torch.float32)

function_signature = {
    "name": "torch_frobenius_norm_bf16",
    "inputs": [
        ((4, 4), torch.float32),
        (None, torch.int32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
