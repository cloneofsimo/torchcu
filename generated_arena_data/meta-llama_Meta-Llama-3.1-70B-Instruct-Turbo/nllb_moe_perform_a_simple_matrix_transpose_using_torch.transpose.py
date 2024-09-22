import torch

def simple_matrix_transpose(matrix: torch.Tensor) -> torch.Tensor:
    """
    This function performs a simple matrix transpose on the given matrix using torch.transpose.
    
    Args:
        matrix (torch.Tensor): The input matrix.
    
    Returns:
        torch.Tensor: The output of the matrix transpose.
    """
    # Perform the matrix transpose
    output_matrix = torch.transpose(matrix, 0, 1)
    
    return output_matrix


function_signature = {
    "name": "simple_matrix_transpose",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}