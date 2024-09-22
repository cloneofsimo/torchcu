
import torch
import torch.nn.functional as F

def avg_pool3d_standardized_weights(input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Applies 3D average pooling to the input tensor and then multiplies the result by standardized weights.
    """
    pooled_output = F.avg_pool3d(input_tensor, kernel_size=3, stride=1, padding=1)
    weights_mean = weights.mean()
    weights_std = weights.std()
    standardized_weights = (weights - weights_mean) / weights_std
    return pooled_output * standardized_weights

function_signature = {
    "name": "avg_pool3d_standardized_weights",
    "inputs": [
        ((2, 3, 4, 5, 6), torch.float32),  # Input tensor with size (batch, channels, depth, height, width)
        ((3, 3, 3), torch.float32),  # Weights tensor with size (kernel_depth, kernel_height, kernel_width)
    ],
    "outputs": [
        ((2, 3, 4, 5, 6), torch.float32),  # Output tensor with same size as input
    ]
}

