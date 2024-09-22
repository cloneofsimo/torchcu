
import torch

def mean_fp16_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate the mean of the input tensor using fp16.
    """
    input_fp16 = input_tensor.to(torch.float16)
    mean = torch.mean(input_fp16)
    return mean.to(torch.float32)

function_signature = {
    "name": "mean_fp16_function",
    "inputs": [
        ((1,), torch.float32),
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
