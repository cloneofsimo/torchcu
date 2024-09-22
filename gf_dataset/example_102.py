
import torch

def torch_identity_fp16(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Perform identity operation on input tensor, return fp16 result.
    """
    return input_tensor.to(torch.float16)

function_signature = {
    "name": "torch_identity_fp16",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float16),
    ]
}
