
import torch

def torch_conv2d_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Perform a 2D convolution with bias using cuDNN.
    """
    output = torch.nn.functional.conv2d(input_tensor, weight, bias=bias, padding=1)
    return output

function_signature = {
    "name": "torch_conv2d_function",
    "inputs": [
        ((1, 3, 28, 28), torch.float32),
        ((3, 3, 3, 3), torch.float32),
        ((3,), torch.float32)
    ],
    "outputs": [
        ((1, 3, 28, 28), torch.float32),
    ]
}
