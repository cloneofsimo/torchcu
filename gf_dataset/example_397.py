
import torch

def torch_linspace_bfloat16_function(start: torch.Tensor, end: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Generate a tensor of evenly spaced values within a given range in bfloat16.
    """
    start_bf16 = start.to(torch.bfloat16)
    end_bf16 = end.to(torch.bfloat16)
    output_bf16 = torch.linspace(start_bf16, end_bf16, steps, dtype=torch.bfloat16)
    return output_bf16.to(torch.float32)

function_signature = {
    "name": "torch_linspace_bfloat16_function",
    "inputs": [
        ((1,), torch.float32),
        ((1,), torch.float32),
        (None, torch.int32)
    ],
    "outputs": [
        ((None,), torch.float32),
    ]
}
