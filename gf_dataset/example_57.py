
import torch

def torch_cholesky_norm_fp16(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Frobenius norm of the Cholesky decomposition of a matrix,
    returning the result in fp16.
    """
    cholesky_decomp = torch.cholesky(input_tensor.to(torch.float32))
    norm = torch.linalg.norm(cholesky_decomp)
    return norm.to(torch.float16)

function_signature = {
    "name": "torch_cholesky_norm_fp16",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((), torch.float16)
    ]
}
