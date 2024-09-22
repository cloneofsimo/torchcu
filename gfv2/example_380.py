
import torch

def my_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations:
    1. Inverts the input tensor.
    2. Calculates the inner product of the inverted tensor with itself.
    3. Applies a sigmoid activation to the result.
    4. Converts the output to fp16.
    """
    inverted_tensor = torch.linalg.inv(input_tensor.to(torch.float32))
    inner_product = torch.matmul(inverted_tensor, inverted_tensor.t())
    output = torch.sigmoid(inner_product)
    return output.to(torch.float16)

function_signature = {
    "name": "my_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float16)
    ]
}
