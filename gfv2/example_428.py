
import torch

def my_complex_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a series of operations on the input tensor, including:
      - Transposing the weight tensor
      - Creating a tensor of ones with the same shape as the input tensor
      - Replicating the padding of the input tensor
      - Performing a matrix multiplication with the transposed weight
      - Applying a sigmoid activation to the result
      - Returning the output and the gradient of the output with respect to the input
    """
    ones_tensor = torch.ones_like(input_tensor, dtype=torch.float32)
    padded_tensor = torch.nn.functional.replication_pad2d(input_tensor, (1, 1, 1, 1))
    
    weight_t = weight.t()
    output = torch.matmul(padded_tensor, weight_t)
    output = torch.sigmoid(output)
    
    # Calculate the gradient of the output with respect to the input
    output.retain_grad()
    output.backward(ones_tensor.to(torch.float16))
    
    return output, input_tensor.grad

function_signature = {
    "name": "my_complex_function",
    "inputs": [
        ((1, 4, 4), torch.float32),  # Input tensor
        ((4, 4), torch.float32)      # Weight tensor
    ],
    "outputs": [
        ((1, 4, 4), torch.float32),  # Output tensor
        ((1, 4, 4), torch.float32)  # Gradient of output with respect to input
    ]
}
