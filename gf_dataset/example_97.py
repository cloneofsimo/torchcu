
import torch

def torch_cumsum_conv2d_function(input_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Performs cumulative sum along a specified dimension, followed by a 2D convolution.
    """
    input_tensor_cumsum = torch.cumsum(input_tensor, dim=1)
    output = torch.nn.functional.conv2d(input_tensor_cumsum.float(), kernel, padding='same')
    return output.int()

function_signature = {
    "name": "torch_cumsum_conv2d_function",
    "inputs": [
        ((1, 3, 5, 5), torch.int8),
        ((1, 1, 3, 3), torch.float32)
    ],
    "outputs": [
        ((1, 3, 5, 5), torch.int8),
    ]
}
