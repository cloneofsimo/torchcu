import torch

def inplace_add(tensor: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    Add a value to a tensor in-place.

    Args:
    tensor (torch.Tensor): The input tensor.
    value (torch.Tensor): The value to add.

    Returns:
    torch.Tensor: The tensor after the in-place addition operation.
    """
    # Perform the in-place addition operation
    tensor.add_(value)

    return tensor



# function_signature
function_signature = {
    "name": "inplace_add",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [((4, 4), torch.float32)]
}