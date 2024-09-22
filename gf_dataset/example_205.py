
import torch
import torch.nn.functional as F

def torch_max_pool_2d_cudnn_function(input_tensor: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """
    Performs 2D max pooling with cuDNN for efficiency.
    """
    output = F.max_pool2d(input_tensor, kernel_size=kernel_size, stride=stride, padding=padding)
    return output

function_signature = {
    "name": "torch_max_pool_2d_cudnn_function",
    "inputs": [
        ((2, 3, 4, 4), torch.float32),  # Example input shape (batch, channels, height, width)
        (1, torch.int32),
        (1, torch.int32),
        (0, torch.int32),
    ],
    "outputs": [
        ((2, 3, 2, 2), torch.float32),  # Example output shape (batch, channels, height, width)
    ]
}
