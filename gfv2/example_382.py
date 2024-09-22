
import torch

def my_complex_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on the input tensor, including:
    - Reshaping
    - Matrix multiplication
    - Element-wise multiplication
    - ReLU activation
    - Summation along a specific dimension
    - Reshaping again
    """
    batch_size = input_tensor.size(0)
    input_tensor = input_tensor.view(batch_size, -1)  # Reshape to (batch_size, feature_dim)
    output = torch.matmul(input_tensor, weight.t())
    output = output * 2.0  # Element-wise multiplication
    output = torch.relu(output)  # ReLU activation
    output = torch.sum(output, dim=1, keepdim=True)  # Sum along the feature dimension
    output = output.view(batch_size, 1, 1, 1)  # Reshape to (batch_size, 1, 1, 1)
    return output

function_signature = {
    "name": "my_complex_function",
    "inputs": [
        ((4, 1, 3, 3), torch.float32),
        ((3, 3), torch.float32)
    ],
    "outputs": [
        ((4, 1, 1, 1), torch.float32),
    ]
}
