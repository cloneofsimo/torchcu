
import torch

def torch_median_ge_function(input_tensor: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """
    Calculate the median of the input tensor and return a tensor of booleans indicating
    whether each element is greater than or equal to the threshold.
    """
    median = torch.median(input_tensor, dim=0)[0]
    return torch.ge(input_tensor, threshold)

function_signature = {
    "name": "torch_median_ge_function",
    "inputs": [
        ((16, 16), torch.float32),
        ((16), torch.float32)
    ],
    "outputs": [
        ((16, 16), torch.bool),
    ]
}
