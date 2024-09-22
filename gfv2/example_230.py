
import torch

def my_function(input_tensor: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Performs a series of operations on the input tensor, including:
    1. Setting a random seed
    2. Adding a constant value
    3. Multiplying by a scalar
    4. Applying a sigmoid function
    5. Returning the result as a float32 tensor
    """
    torch.manual_seed(seed)
    result = input_tensor + 2.0
    result *= 1.5
    result = torch.sigmoid(result)
    return result.to(torch.float32)

function_signature = {
    "name": "my_function",
    "inputs": [
        ((10,), torch.float32),
        (int, )
    ],
    "outputs": [
        ((10,), torch.float32),
    ]
}
