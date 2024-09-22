
import torch
import torch.nn.functional as F

def complex_function(input_tensor: torch.Tensor, 
                     weight: torch.Tensor, 
                     bias: torch.Tensor, 
                     coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a complex series of operations on input tensor:
    1. Coordinate convolution: Applies a convolution based on spatial coordinates.
    2. Linear transformation with bias.
    3. Batch normalization with int8 precision.
    4. ReLU activation.
    5. SVD decomposition and returns the singular values and vectors.

    Args:
        input_tensor: Input tensor of shape (B, C, H, W).
        weight: Convolutional kernel of shape (Cout, Cin, KH, KW).
        bias: Bias tensor of shape (Cout).
        coords: Coordinate tensor of shape (B, 2, H, W).

    Returns:
        A tuple containing:
            - Singular values of the output tensor.
            - Singular vectors of the output tensor.
    """
    # Coordinate convolution
    B, C, H, W = input_tensor.shape
    coord_conv_output = F.conv2d(input_tensor, weight, bias=bias, padding="same")
    
    # Batch normalization with int8 precision
    coord_conv_output = coord_conv_output.to(torch.int8)
    bn_output = torch.nn.functional.batch_norm(coord_conv_output, eps=1e-5, momentum=0.1, affine=True, training=False)
    bn_output = bn_output.to(torch.bfloat16)
    
    # ReLU activation
    relu_output = F.relu(bn_output, inplace=True)
    
    # SVD decomposition
    U, S, V = torch.linalg.svd(relu_output)

    return S, V

function_signature = {
    "name": "complex_function",
    "inputs": [
        ((4, 3, 16, 16), torch.float32),
        ((4, 3, 3, 3), torch.float32),
        ((4,), torch.float32),
        ((4, 2, 16, 16), torch.float32)
    ],
    "outputs": [
        ((4,), torch.float32),
        ((4, 4, 16, 16), torch.float32)
    ]
}
