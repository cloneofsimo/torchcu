import torch

def simple_matrix_multiplication(matrix1: torch.Tensor, matrix2: torch.Tensor) -> torch.Tensor:
    """
    This function performs a simple matrix multiplication on the given matrices using torch.matmul.
    
    Args:
        matrix1 (torch.Tensor): The first matrix.
        matrix2 (torch.Tensor): The second matrix.
    
    Returns:
        torch.Tensor: The output of the matrix multiplication.
    """
    # Perform the matrix multiplication
    output_matrix = torch.matmul(matrix1, matrix2)
    
    return output_matrix


function_signature = {
    "name": "simple_matrix_multiplication",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}