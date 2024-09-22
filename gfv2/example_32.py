
import torch
import torch.nn.functional as F

def unique_and_sum(input_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Finds unique elements in the input tensor and sums the corresponding values in the mask.
    """
    # Find unique elements in the input tensor
    unique_vals, inverse_indices = torch.unique(input_tensor, return_inverse=True)

    # Create a tensor of ones with the same shape as the input tensor
    ones_tensor = torch.ones_like(input_tensor)

    # Use scatter_add to sum the values in the mask tensor based on the unique elements
    summed_mask = torch.zeros_like(unique_vals, dtype=torch.float32)
    torch.scatter_add(summed_mask, 0, inverse_indices, ones_tensor * mask)

    return summed_mask

function_signature = {
    "name": "unique_and_sum",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4,), torch.float32),
    ]
}
