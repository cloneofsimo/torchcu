
import torch

def max_pool_fp32_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Perform a 2D max pooling operation with a kernel size of 2 and stride of 2,
    returning the result in FP32.
    """
    output = torch.nn.functional.max_pool2d(input_tensor, kernel_size=2, stride=2)
    return output.to(torch.float32)

function_signature = {
    "name": "max_pool_fp32_function",
    "inputs": [
        ((2, 3, 4, 4), torch.float32)
    ],
    "outputs": [
        ((2, 3, 2, 2), torch.float32)
    ]
}
