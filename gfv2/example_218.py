
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(MyModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        x = self.conv(x)
        x = torch.sigmoid(x)  
        return x

def my_function(input_tensor: torch.Tensor, list_of_tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Performs a series of operations on the input tensor and a list of tensors.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        list_of_tensors (list[torch.Tensor]): A list of tensors.

    Returns:
        torch.Tensor: The output tensor.
    """

    # Hadamard product with the first tensor in the list
    x = torch.mul(input_tensor, list_of_tensors[0])
    
    # Element-wise maximum with the second tensor in the list
    x = torch.max(x, list_of_tensors[1])
    
    # Eigenvalue decomposition and keep the largest eigenvalue
    x = torch.linalg.eigvals(x)
    x = torch.max(x, dim=1).values
    
    # Separable convolution with a custom kernel
    conv_module = MyModule(x.shape[1], 4, 3)
    x = conv_module(x.unsqueeze(1).unsqueeze(1))
    
    # Apply ReLU activation and convert to int8
    x = F.relu(x)
    x = x.to(torch.int8)
    
    # Inplace addition with the third tensor in the list
    x.add_(list_of_tensors[2])
    
    # Return the result as a float32 tensor
    return x.to(torch.float32)

function_signature = {
    "name": "my_function",
    "inputs": [
        ((1, 16, 16, 16), torch.float32),
        [((1, 16, 16, 16), torch.float32), ((1, 16, 16, 16), torch.float32), ((1, 4, 16, 16), torch.float32)]
    ],
    "outputs": [
        ((1, 4, 16, 16), torch.float32)
    ]
}
