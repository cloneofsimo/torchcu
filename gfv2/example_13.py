
import torch

def torch_baddbmm_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a batched matrix multiplication (bmm) followed by a batched addition (badd) with bias.
    """
    output = torch.baddbmm(input_tensor, weight, bias)
    return output

function_signature = {
    "name": "torch_baddbmm_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4, 4), torch.float32),
    ]
}
