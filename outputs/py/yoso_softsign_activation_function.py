import torch

def softsign_activation(tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply the softsign activation function to a given tensor.

    Args:
    tensor (torch.Tensor): The input tensor.

    Returns:
    torch.Tensor: The output tensor after applying the softsign activation function.
    """
    # Apply the softsign activation function
    output = tensor / (1 + torch.abs(tensor))

    return output



# function_signature
function_signature = {
    "name": "softsign_activation",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [((4, 4), torch.float32)]
}