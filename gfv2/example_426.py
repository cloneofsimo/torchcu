
import torch

def my_complex_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a complex series of operations:
    1. Matrix multiplication with weight
    2. Element-wise multiplication with another tensor (created with zeros_like)
    3. Applies sigmoid activation
    4. Computes the mean along a specific dimension
    5. Returns the result and also a tensor of ones like the input tensor
    """
    output = torch.matmul(input_tensor, weight.t())
    zero_tensor = torch.zeros_like(output)
    output = torch.mul(output, zero_tensor + 1.0)  # Element-wise multiplication with ones
    output = torch.sigmoid(output)
    output_mean = torch.mean(output, dim=1, keepdim=True)
    ones_tensor = torch.ones_like(input_tensor)
    return output_mean, ones_tensor

function_signature = {
    "name": "my_complex_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 1), torch.float32),
        ((4, 4), torch.float32)
    ]
}
