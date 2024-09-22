
import torch

def torch_max_filter_fp16_function(input_tensor: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """
    Performs a max pooling operation with specified kernel size and stride, using fp16 precision.
    """
    input_fp16 = input_tensor.to(torch.float16)
    output = torch.nn.functional.max_pool2d(input_fp16, kernel_size=kernel_size, stride=stride)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_max_filter_fp16_function",
    "inputs": [
        ((1, 1, 4, 4), torch.float32),
        (1, torch.int32),
        (1, torch.int32)
    ],
    "outputs": [
        ((1, 1, 2, 2), torch.float32),
    ]
}
