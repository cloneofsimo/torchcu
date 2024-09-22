
import torch

def linspace_floor_fp16(input_tensor: torch.Tensor, start: float, end: float, steps: int) -> torch.Tensor:
    """
    Generates a tensor of evenly spaced values within a given range, rounds down each element, and converts the tensor to fp16.
    """
    linspace_tensor = torch.linspace(start, end, steps)
    floored_tensor = torch.floor(linspace_tensor)
    return floored_tensor.to(torch.float16)

function_signature = {
    "name": "linspace_floor_fp16",
    "inputs": [
        ((1,), torch.float32),
        (float,),
        (float,),
        (int,)
    ],
    "outputs": [
        ((1,), torch.float16),
    ]
}
