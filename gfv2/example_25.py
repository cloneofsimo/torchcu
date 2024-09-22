
import torch
import torch.nn.functional as F

def torch_box_filter_function(input_tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Applies a box filter (average pooling) to the input tensor.

    Args:
        input_tensor: The input tensor to filter.
        kernel_size: The size of the box filter kernel.

    Returns:
        The filtered tensor.
    """
    return F.avg_pool2d(input_tensor, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

function_signature = {
    "name": "torch_box_filter_function",
    "inputs": [
        ((1, 3, 32, 32), torch.float32),
        (1, torch.int32),
    ],
    "outputs": [
        ((1, 3, 32, 32), torch.float32),
    ]
}
