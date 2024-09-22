
import torch

def my_complex_function(input_tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Performs a series of operations on the input tensor:
        1. Scales the input tensor by the given scale.
        2. Applies a sigmoid activation function.
        3. Performs element-wise multiplication with a uniform random tensor.
        4. Calculates the mean of each row.
    """
    scaled_input = input_tensor * scale
    activated_input = torch.sigmoid(scaled_input)
    random_tensor = torch.rand_like(activated_input, dtype=torch.float32)
    output = activated_input * random_tensor
    row_means = torch.mean(output, dim=1, keepdim=True)
    return row_means

function_signature = {
    "name": "my_complex_function",
    "inputs": [
        ((4, 4), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((4, 1), torch.float32)
    ]
}
