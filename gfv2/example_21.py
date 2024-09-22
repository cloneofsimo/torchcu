
import torch
import torch.nn.functional as F

def dilation_mask_attention_int8(input_tensor: torch.Tensor, kernel: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Performs morphological dilation followed by mask-based attention using int8 precision.
    """
    # 1. Morphological Dilation
    dilated = F.max_pool2d(input_tensor.to(torch.int8), kernel_size=kernel.shape, stride=1, padding=kernel.shape[0] // 2)

    # 2. Mask-based Attention
    masked_dilated = dilated * mask.to(torch.int8)

    # 3. Convert back to float32
    return masked_dilated.to(torch.float32)

function_signature = {
    "name": "dilation_mask_attention_int8",
    "inputs": [
        ((1, 1, 32, 32), torch.float32),
        ((3, 3), torch.int64),
        ((1, 1, 32, 32), torch.float32)
    ],
    "outputs": [
        ((1, 1, 32, 32), torch.float32),
    ]
}
