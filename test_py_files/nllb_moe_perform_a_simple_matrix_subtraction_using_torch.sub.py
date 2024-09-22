import torch

def simple_matrix_subtraction(matrix1: torch.Tensor, matrix2: torch.Tensor) -> torch.Tensor:
    """
    This function performs a simple matrix subtraction on the given matrices using torch.sub.
    
    Args:
        matrix1 (torch.Tensor): The first matrix.
        matrix2 (torch.Tensor): The second matrix.
    
    Returns:
        torch.Tensor: The output of the matrix subtraction.
    """
    # Perform the matrix subtraction
    output_matrix = torch.sub(matrix1, matrix2)
    
    return output_matrix


function_signature = {
    "name": "simple_matrix_subtraction",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}