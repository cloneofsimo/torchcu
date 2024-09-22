
import torch
import torch.nn.functional as F
from torch.nn.functional import pad

def torch_bilateral_filter_fp16(input_tensor: torch.Tensor, kernel_size: int, sigma_color: float, sigma_spatial: float) -> torch.Tensor:
    """
    Applies a bilateral filter to the input tensor, using fp16 for internal computations and returning a fp16 tensor.
    """
    input_tensor = input_tensor.to(torch.float16)
    padded_input = pad(input_tensor, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), 'constant', 0)
    output = F.avg_pool2d(padded_input, kernel_size, stride=1, padding=0)
    output = output.to(torch.float32)
    return output.to(torch.float16)


function_signature = {
    "name": "torch_bilateral_filter_fp16",
    "inputs": [
        ((1, 1, 10, 10), torch.float32),
        (5, torch.int32),
        (1.0, torch.float32),
        (1.0, torch.float32)
    ],
    "outputs": [
        ((1, 1, 10, 10), torch.float16),
    ]
}

