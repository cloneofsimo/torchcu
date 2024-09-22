
import torch

def my_complex_function(input_tensor: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations, including view, matrix multiplication, and element-wise addition.
    """
    # View the input tensor to reshape it
    input_tensor = input_tensor.view(input_tensor.size(0), -1)
    # Perform matrix multiplication with weight1
    output1 = torch.matmul(input_tensor, weight1.t())
    # Perform matrix multiplication with weight2
    output2 = torch.matmul(output1, weight2.t())
    # Add output1 and output2 element-wise
    output = output1 + output2
    return output

function_signature = {
    "name": "my_complex_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
