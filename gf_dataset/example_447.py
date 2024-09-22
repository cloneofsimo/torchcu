
import torch
import torch.nn.functional as F

def torch_maxpool_bf16_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a 2D max pooling operation with bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output = F.max_pool2d(input_bf16, kernel_size=2, stride=2)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_maxpool_bfloat16_function",
    "inputs": [
        ((2, 3, 4, 4), torch.float32),
    ],
    "outputs": [
        ((2, 3, 2, 2), torch.float32),
    ]
}
