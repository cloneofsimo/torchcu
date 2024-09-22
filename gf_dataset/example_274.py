
import torch

def torch_cholesky_bfloat16_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Cholesky decomposition of a positive-definite matrix using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output_bf16 = torch.linalg.cholesky(input_bf16)
    return output_bf16.to(torch.float32)

function_signature = {
    "name": "torch_cholesky_bfloat16_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
