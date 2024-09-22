
import torch

def my_function(input_tensor: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a linear transformation with weight standardization, applies ReLU activation, and returns the result.
    """
    # Weight standardization
    weight_mean = torch.mean(weights, dim=1, keepdim=True)
    weight_std = torch.std(weights, dim=1, keepdim=True)
    standardized_weights = (weights - weight_mean) / weight_std

    # Linear transformation
    output = torch.matmul(input_tensor, standardized_weights.t()) + bias

    # ReLU activation
    output = torch.relu(output)

    return output

function_signature = {
    "name": "my_function",
    "inputs": [
        ((16, 4), torch.float32),
        ((4, 16), torch.float32),
        ((16,), torch.float32)
    ],
    "outputs": [
        ((16, 16), torch.float32),
    ]
}
