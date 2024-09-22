
import torch

def torch_std_fp16_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate the standard deviation of the input tensor in FP16.
    """
    input_fp16 = input_tensor.to(torch.float16)
    std = torch.std(input_fp16, dim=0, keepdim=True)
    return std.to(torch.float32)

function_signature = {
    "name": "torch_std_fp16_function",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((1, 4), torch.float32),
    ]
}
