
import torch
import torch.nn as nn

def coord_attention_fp32(input_tensor: torch.Tensor, k: int) -> torch.Tensor:
    """
    Applies Coord Attention to an input tensor.

    Args:
        input_tensor: The input tensor.
        k: The kernel size for the convolution operation.

    Returns:
        The output tensor after applying Coord Attention.
    """

    # Original Implementation - Not the most efficient but demonstrates the core logic:
    B, C, H, W = input_tensor.size()

    # Generate coordinate maps
    x_range = torch.arange(W).float().view(1, 1, 1, W).expand(B, C, H, W).to(input_tensor.device)
    y_range = torch.arange(H).float().view(1, 1, H, 1).expand(B, C, H, W).to(input_tensor.device)
    
    # Normalize coordinates
    x_range = (x_range - W / 2) / (W / 2)
    y_range = (y_range - H / 2) / (H / 2)

    # Concatenate coordinates to the input
    input_with_coords = torch.cat([input_tensor, x_range, y_range], dim=1)  

    # Apply convolution
    conv_output = nn.Conv2d(C + 2, C, kernel_size=k, padding=k//2, bias=False)(input_with_coords)

    # Calculate attention weights (Original implementation using FC layers)
    attention_weights = nn.Linear(C, 1)(conv_output.view(B, C, -1)).view(B, 1, H, W) 
    attention_weights = torch.sigmoid(attention_weights)

    # Apply attention weights
    output_tensor = input_tensor * attention_weights

    return output_tensor

function_signature = {
    "name": "coord_attention_fp32",
    "inputs": [
        ((16, 64, 32, 32), torch.float32),
        (3, torch.int32)
    ],
    "outputs": [
        ((16, 64, 32, 32), torch.float32),
    ]
}
