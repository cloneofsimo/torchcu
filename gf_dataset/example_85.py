
import torch
import torch.nn.functional as F
from torch.nn.functional import morphology

def torch_erosion_sort_bmm_function(input_tensor: torch.Tensor, kernel: torch.Tensor, other_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs morphological erosion, sorts the output, then performs a batch matrix multiplication. 
    """
    eroded_output = morphology.erosion(input_tensor, kernel)
    sorted_output, _ = torch.sort(eroded_output, dim=1, descending=True)
    bmm_output = torch.bmm(sorted_output, other_tensor)
    return bmm_output.to(torch.float32)

function_signature = {
    "name": "torch_erosion_sort_bmm_function",
    "inputs": [
        ((16, 16, 32), torch.float32),  # Input tensor
        ((3, 3), torch.float32),  # Kernel for erosion
        ((32, 16), torch.float32),  # Other tensor for batch matrix multiplication
    ],
    "outputs": [
        ((16, 16, 16), torch.float32),
    ]
}
