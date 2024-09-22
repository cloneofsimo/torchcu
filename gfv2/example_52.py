
import torch
import torch.nn.functional as F

def max_pool_bf16_fading(input_tensor: torch.Tensor, kernel_size: int, stride: int, fading_factor: float) -> torch.Tensor:
    """
    Applies a 2D max pooling operation using bfloat16, followed by fading out.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    pooled = F.max_pool2d(input_bf16, kernel_size=kernel_size, stride=stride)
    output = pooled * (1 - fading_factor)  # Fading out
    return output.to(torch.float32)

function_signature = {
    "name": "max_pool_bf16_fading",
    "inputs": [
        ((4, 3, 16, 16), torch.float32),
        (2, torch.int32),
        (2, torch.int32),
        (torch.float32)
    ],
    "outputs": [
        ((4, 3, 8, 8), torch.float32)
    ]
}
