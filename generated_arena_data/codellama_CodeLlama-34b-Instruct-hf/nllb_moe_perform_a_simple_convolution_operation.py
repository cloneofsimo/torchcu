import torch

def simple_convolution(input_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    This function performs a simple convolution operation on the input tensor using the given kernel.
    
    Args:
        input_tensor (torch.Tensor): The input tensor to be convolved.
        kernel (torch.Tensor): The kernel to be used for convolution.
    
    Returns:
        torch.Tensor: The output of the convolution operation.
    """
    # Get the dimensions of the input tensor and the kernel
    input_height, input_width, input_channels = input_tensor.shape
    kernel_height, kernel_width, kernel_channels = kernel.shape
    
    # Initialize the output tensor with zeros
    output_tensor = torch.zeros((input_height, input_width, kernel_channels))
    
    # Perform the convolution operation
    for i in range(input_height):
        for j in range(input_width):
            for k in range(input_channels):
                for m in range(kernel_height):
                    for n in range(kernel_width):
                        output_tensor[i, j, :] += input_tensor[i, j, k] * kernel[m, n, k]
    
    return output_tensor


function_signature = {
    "name": "simple_convolution",
    "inputs": [
        ((4, 4, 4), torch.float32),
        ((4, 4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4, 4), torch.float32),
    ]
}
