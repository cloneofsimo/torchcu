import torch

def forward_function(tensor: torch.Tensor) -> torch.Tensor:
    """
    A simple forward function that takes a tensor as input and returns the tensor itself.

    Args:
    tensor (torch.Tensor): The input tensor.

    Returns:
    torch.Tensor: The output tensor.
    """
    # Perform the forward operation
    output = tensor

    return output



# function_signature
function_signature = {
    "name": "forward_function",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [((4, 4), torch.float32)]
}