
import torch

def torch_sigmoid_arange_bfloat16_function(input_tensor: torch.Tensor, start: int, end: int, step: int) -> torch.Tensor:
    """
    Calculate sigmoid of a range of numbers using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    arange_bf16 = torch.arange(start, end, step, dtype=torch.bfloat16)
    output = torch.sigmoid(input_bf16 + arange_bf16)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_sigmoid_arange_bfloat16_function",
    "inputs": [
        ((1,), torch.float32),
        ((), torch.int32),
        ((), torch.int32),
        ((), torch.int32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
