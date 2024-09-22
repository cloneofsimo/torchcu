import torch
import numpy as np

def convolution_with_int8_precision(input_tensor: torch.Tensor, kernel_tensor: torch.Tensor) -> torch.Tensor:
    """
    This function performs a convolution operation on the input tensor using the given kernel tensor with int8 precision.
    
    Args:
    input_tensor (torch.Tensor): The input tensor to be convolved.
    kernel_tensor (torch.Tensor): The kernel tensor used for convolution.
    
    Returns:
    torch.Tensor: The output tensor after convolution.
    """
    # Get the dimensions of the input tensor and kernel tensor
    input_dim = input_tensor.shape
    kernel_dim = kernel_tensor.shape
    
    # Calculate the output dimensions
    output_dim = (input_dim[0] - kernel_dim[0] + 1, input_dim[1] - kernel_dim[1] + 1, input_dim[2] - kernel_dim[2] + 1)
    
    # Initialize the output tensor with zeros
    output_tensor = torch.zeros(output_dim, dtype=torch.int8)
    
    # Perform the convolution operation
    for i in range(output_dim[0]):
        for j in range(output_dim[1]):
            for k in range(output_dim[2]):
                output_tensor[i, j, k] = torch.sum(input_tensor[i:i+kernel_dim[0], j:j+kernel_dim[1], k:k+kernel_dim[2]] * kernel_tensor)
    
    return output_tensor



# function_signature
function_signature = {
    "name": "convolution_with_int8_precision",
    "inputs": [
        ((4, 4, 4), torch.float32),
        ((4, 4, 4), torch.float32)
    ],
    "outputs": [((1, 1, 1), torch.int8)]
}