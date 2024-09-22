
import torch

def baddbmm_function(input_tensor: torch.Tensor, batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
    """
    Performs a batch matrix multiplication of matrices in batch1 and batch2, 
    adds the result to input_tensor, and returns the output.
    """
    return torch.baddbmm(input_tensor, batch1, batch2)

function_signature = {
    "name": "baddbmm_function",
    "inputs": [
        ((1, 4, 4), torch.float32), 
        ((2, 4, 4), torch.float32),
        ((2, 4, 4), torch.float32)
    ],
    "outputs": [
        ((1, 4, 4), torch.float32),
    ]
}
