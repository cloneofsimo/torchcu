import torch

def simple_matrix_multiplication(matrix1: torch.Tensor, matrix2: torch.Tensor) -> torch.Tensor:
    """
    This function performs a simple matrix multiplication on the given matrices.
    
    Args:
        matrix1 (torch.Tensor): The first matrix.
        matrix2 (torch.Tensor): The second matrix.
    
    Returns:
        torch.Tensor: The output of the matrix multiplication.
    """
    # Get the dimensions of the matrices
    num_rows_matrix1, num_cols_matrix1 = matrix1.shape
    num_rows_matrix2, num_cols_matrix2 = matrix2.shape
    
    # Check if the matrices can be multiplied
    if num_cols_matrix1 != num_rows_matrix2:
        raise ValueError("The number of columns in the first matrix must be equal to the number of rows in the second matrix.")
    
    # Initialize the output matrix with zeros
    output_matrix = torch.zeros((num_rows_matrix1, num_cols_matrix2))
    
    # Perform the matrix multiplication
    for i in range(num_rows_matrix1):
        for j in range(num_cols_matrix2):
            for k in range(num_cols_matrix1):
                output_matrix[i, j] += matrix1[i, k] * matrix2[k, j]
    
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