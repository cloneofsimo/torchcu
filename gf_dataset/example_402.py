
import torch
import torch.nn.functional as F

def torch_log_box_filter(input_tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Applies a box filter to the input tensor, then takes the logarithm of the result.
    This operation is applied in-place.
    """
    input_tensor.log_()  # In-place logarithm
    F.avg_pool2d(input_tensor, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False)
    return input_tensor

function_signature = {
    "name": "torch_log_box_filter",
    "inputs": [
        ((2, 3, 5, 5), torch.float32),
        (3, torch.int32)  # kernel_size
    ],
    "outputs": [
        ((2, 3, 5, 5), torch.float32),
    ]
}
