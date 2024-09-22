
import torch

def repeat_tensor_fp16(input_tensor: torch.Tensor, repeat_times: int) -> torch.Tensor:
    """
    Repeats a tensor along the first dimension for a given number of times.
    The input tensor is converted to fp16 for faster computation.
    """
    input_fp16 = input_tensor.to(torch.float16)
    output = input_fp16.repeat(repeat_times, 1, 1)
    return output.to(torch.float32)

function_signature = {
    "name": "repeat_tensor_fp16",
    "inputs": [
        ((1, 2, 3), torch.float32),
        ((), torch.int32),
    ],
    "outputs": [
        ((3, 2, 3), torch.float32),
    ]
}
