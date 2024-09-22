
import torch

def complex_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> list:
    """
    This function performs a series of operations, including eigenvalue calculation,
    weight standardization, and element-wise comparisons, returning both the
    result of the comparisons and a standardized weight tensor.

    Args:
        input_tensor: A tensor of size at least 1.
        weight: A tensor of size at least 1.

    Returns:
        A list containing two tensors:
            - A tensor with the result of element-wise comparisons.
            - A standardized weight tensor.
    """
    
    # Calculate eigenvalues of the input tensor
    eigenvalues, _ = torch.linalg.eig(input_tensor.float())
    eigenvalues = eigenvalues.squeeze()

    # Standardize the weight tensor
    weight_mean = weight.mean()
    weight_std = weight.std()
    standardized_weight = (weight - weight_mean) / weight_std

    # Perform element-wise comparisons
    comparison_result = (eigenvalues > 0).any()

    return [comparison_result.float(), standardized_weight.float()]

function_signature = {
    "name": "complex_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
        ((4, 4), torch.float32)
    ]
}
