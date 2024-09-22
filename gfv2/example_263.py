
import torch

def my_function(input_tensor: torch.Tensor,  output_size: int) -> torch.Tensor:
    """
    Performs a series of operations on a tensor, including transposition,
    arange, and in-place operations, before returning the result.
    """
    # Transpose the input tensor
    transposed = input_tensor.t()

    # Create a range tensor
    arange_tensor = torch.arange(output_size, dtype=torch.float32)

    # Multiply the transposed tensor with the range tensor and perform in-place addition
    transposed.mul_(arange_tensor)

    # Return the result
    return transposed

function_signature = {
    "name": "my_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((), torch.int32)  # output_size as a scalar
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
