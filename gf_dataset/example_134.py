
import torch

def torch_clamp_fp16_function(input_tensor: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    """
    Clamps the input tensor to the given range and returns the result in fp16.
    """
    input_fp16 = input_tensor.to(torch.float16)
    output = torch.clamp(input_fp16, min_value, max_value)
    return output

function_signature = {
    "name": "torch_clamp_fp16_function",
    "inputs": [
        ((4, 4), torch.float32),
        (torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float16),
    ]
}
