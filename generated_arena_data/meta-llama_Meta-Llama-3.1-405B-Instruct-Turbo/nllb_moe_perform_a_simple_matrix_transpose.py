import torch

def simple_matrix_transpose(matrix: torch.Tensor) -> torch.Tensor:
    """
    This function performs a simple matrix transpose on the given matrix.
    
    Args:
        matrix (torch.Tensor): The input matrix.
    
    Returns:
        torch.Tensor: The output of the matrix transpose.
    """
    # Get the dimensions of the matrix
    num_rows, num_cols = matrix.shape
    
    # Initialize the output matrix with zeros
    output_matrix = torch.zeros((num_cols, num_rows))
    
    # Perform the matrix transpose
    for i in range(num_rows):
        for j in range(num_cols):
            output_matrix[j, i] = matrix[i, j]
    
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