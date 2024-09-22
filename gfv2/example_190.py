
import torch

def fading_in_matmul_fp32_function(input_tensor: torch.Tensor, weight: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Performs a matrix multiplication with fading-in factor.
    """
    input_fp32 = input_tensor.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    output = torch.matmul(input_fp32, weight_fp32.t())
    return (1 - alpha) * output + alpha * input_tensor

function_signature = {
    "name": "fading_in_matmul_fp32_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
