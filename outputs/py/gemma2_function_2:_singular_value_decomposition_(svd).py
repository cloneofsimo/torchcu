import torch
import torch.linalg

def svd(matrix: torch.Tensor) -> torch.Tensor:
    """
    Performs the singular value decomposition (SVD) on the given matrix.

    Args:
        matrix (torch.Tensor): The input matrix.

    Returns:
        torch.Tensor: The U matrix of the SVD decomposition.
    """
    # Perform the SVD decomposition
    u, s, vh = torch.linalg.svd(matrix)

    # Return the U matrix
    return u



# function_signature
function_signature = {
    "name": "svd",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [((4, 4), torch.float32)]
}