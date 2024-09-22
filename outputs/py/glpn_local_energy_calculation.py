import torch

def local_energy(tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate the local energy of a tensor.

    Args:
    tensor (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: Local energy of the input tensor.
    """
    # Calculate the square of the tensor
    squared_tensor = tensor ** 2

    # Calculate the mean of the squared tensor along the last axis
    local_energy = torch.mean(squared_tensor, dim=-1)

    return local_energy



# function_signature
function_signature = {
    "name": "local_energy",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [((4), torch.float32)]
}