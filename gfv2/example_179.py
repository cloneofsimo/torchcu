
import torch
import numpy as np

def inverse_discrete_wavelet_transform_double_linear_addr_inplace_backward_fp32(
    input_tensor: torch.Tensor, 
    wavelet: str, 
    mode: str, 
    weights1: torch.Tensor, 
    weights2: torch.Tensor
) -> torch.Tensor:
    """
    Applies an inverse discrete wavelet transform (IDWT) to the input tensor,
    followed by two linear layers, a pointwise addition, and an inplace backward pass.
    
    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        wavelet (str): Wavelet name (e.g., 'db4').
        mode (str): Mode of the wavelet transform (e.g., 'symmetric').
        weights1 (torch.Tensor): Weights for the first linear layer.
        weights2 (torch.Tensor): Weights for the second linear layer.

    Returns:
        torch.Tensor: Output tensor after applying the operations.
    """
    
    # Apply inverse wavelet transform
    output_tensor = torch.from_numpy(np.array([
        pywt.idwt2(input_tensor[i].numpy(), wavelet, mode)
        for i in range(input_tensor.shape[0])
    ])).float()
    
    # Apply two linear layers
    output_tensor = torch.nn.functional.linear(output_tensor, weights1)
    output_tensor = torch.nn.functional.linear(output_tensor, weights2)
    
    # Add with input tensor (pointwise addition)
    output_tensor.add_(input_tensor)
    
    # Apply backward pass
    output_tensor.backward()
    
    # Return the output tensor with float32 precision
    return output_tensor.float()

function_signature = {
    "name": "inverse_discrete_wavelet_transform_double_linear_addr_inplace_backward_fp32",
    "inputs": [
        ((1, 1, 32, 32), torch.float32),
        ("db4", torch.float32),
        ("symmetric", torch.float32),
        ((10, 1024), torch.float32),
        ((10, 1024), torch.float32)
    ],
    "outputs": [
        ((1, 1, 32, 32), torch.float32)
    ]
}
