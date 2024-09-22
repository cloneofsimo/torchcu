
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(MyModule, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1) 
        return x

def my_function(input_tensor: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor) -> tuple:
    """
    Performs a series of operations on input tensors, including:
        - Transposed 3D convolution
        - ReLU activation
        - View operation to flatten
        - Center loss calculation
        - K-th value retrieval
        - Int8 conversion

    Args:
        input_tensor: Input tensor of shape [batch_size, in_channels, D, H, W]
        weight: Weight tensor for the transposed convolution
        labels: Labels for the center loss calculation

    Returns:
        Tuple containing:
            - Output tensor of shape [batch_size, out_channels * D * H * W]
            - Center loss value
            - K-th value from the output tensor
    """

    # Create a module instance for the transposed convolution
    module = MyModule(in_channels=input_tensor.shape[1], out_channels=weight.shape[1], kernel_size=weight.shape[2:])

    # Forward pass through the module
    output = module(input_tensor)

    # Center loss calculation
    center_loss = F.cross_entropy(output, labels)

    # K-th value retrieval
    kth_value = torch.kthvalue(output, k=2, dim=1)[0]

    # Int8 conversion
    output_int8 = output.to(torch.int8)

    return output_int8, center_loss, kth_value

function_signature = {
    "name": "my_function",
    "inputs": [
        ((1, 2, 4, 4, 4), torch.float32),
        ((2, 2, 3, 3, 3), torch.float32),
        ((1,), torch.int64)
    ],
    "outputs": [
        ((1, 288), torch.int8),
        ((), torch.float32),
        ((1,), torch.float32)
    ]
}
