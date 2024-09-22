import torch

def batch_matrix_multiply(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Perform batch matrix multiplication.

    Args:
    tensor1 (torch.Tensor): First tensor.
    tensor2 (torch.Tensor): Second tensor.

    Returns:
    torch.Tensor: Result of batch matrix multiplication.
    """
    # Perform batch matrix multiplication using torch.bmm
    result = torch.bmm(tensor1, tensor2)

    return result



# function_signature
function_signature = {
    "name": "batch_matrix_multiply",
    "inputs": [
        ((4, 4, 4), torch.float32),
        ((4, 4, 4), torch.float32)
    ],
    "outputs": [((4, 4, 4), torch.float32)]
}