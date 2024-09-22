
import torch

def torch_sort_median_fp32_function(input_tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Sorts the input tensor along the specified dimension and returns the median value.
    """
    sorted_tensor = torch.sort(input_tensor, dim=dim)[0]  # Get sorted values
    median_value = torch.median(sorted_tensor, dim=dim)[0]  # Get median value along the specified dim
    return median_value.to(torch.float32)  # Return median value in fp32

function_signature = {
    "name": "torch_sort_median_fp32_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
