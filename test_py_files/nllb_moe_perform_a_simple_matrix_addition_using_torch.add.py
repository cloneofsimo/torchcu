import torch

def simple_matrix_addition(matrix1: torch.Tensor, matrix2: torch.Tensor) -> torch.Tensor:
    """
    This function performs a simple matrix addition on the given matrices using torch.add.
    
    Args:
        matrix1 (torch.Tensor): The first matrix.
        matrix2 (torch.Tensor): The second matrix.
    
    Returns:
        torch.Tensor: The output of the matrix addition.
    """
    # Perform the matrix addition
    output_matrix = torch.add(matrix1, matrix2)
    
    return output_matrix


function_signature = {
    "name": "simple_matrix_addition",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}