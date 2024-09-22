
import torch
import torch.nn.functional as F

def mean_maxpool_fp16_function(input_tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Calculates the mean of the input tensor, performs max pooling, and returns the result in fp16.
    """
    input_fp16 = input_tensor.to(torch.float16)
    mean_val = torch.mean(input_fp16)
    output = F.max_pool2d(input_fp16, kernel_size=kernel_size)
    return output.to(torch.float16), mean_val.to(torch.float16)

function_signature = {
    "name": "mean_maxpool_fp16_function",
    "inputs": [
        ((1, 1, 4, 4), torch.float32),
        (2, )
    ],
    "outputs": [
        ((1, 1, 2, 2), torch.float16),
        ((), torch.float16)
    ]
}
