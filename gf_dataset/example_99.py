
import torch

def torch_fold_clamp_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a folding operation, clamps the result, and returns the output.
    """
    output = torch.add(torch.matmul(input_tensor, weight), bias)
    output = torch.clamp(output, min=-1.0, max=1.0)
    return output

function_signature = {
    "name": "torch_fold_clamp_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 1), torch.float32)
    ],
    "outputs": [
        ((4, 1), torch.float32)
    ]
}
