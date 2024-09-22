
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def torch_sparse_conv(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Sparse convolution operation with optional bias.
    
    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        weight (torch.Tensor): Sparse weight tensor of shape (out_channels, in_channels, kernel_size, kernel_size),
                               where each entry represents the sparsity of the corresponding weight.
        bias (torch.Tensor): Optional bias tensor of shape (out_channels).
        indices (torch.Tensor): Indices tensor for sparse weights, indicating non-zero weights.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
    """
    
    # Apply sparse weights
    output = torch.sparse.mm(input_tensor.view(input_tensor.shape[0], -1), weight.t()).view(input_tensor.shape[0], weight.shape[0], input_tensor.shape[2], input_tensor.shape[3])
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    # Apply ReLU activation
    output = F.relu(output)
    return output

function_signature = {
    "name": "torch_sparse_conv",
    "inputs": [
        ((1, 3, 32, 32), torch.float32),
        ((16, 3, 3, 3), torch.float32),
        ((16,), torch.float32),
        ((16, 3, 3, 3), torch.bool)
    ],
    "outputs": [
        ((1, 16, 32, 32), torch.float32),
    ]
}
