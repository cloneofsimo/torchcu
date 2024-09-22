
import torch

def masked_select_fp32(input_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Applies a mask to select elements from the input tensor and returns the result in float32.
    """
    masked_tensor = torch.masked_select(input_tensor, mask)
    return masked_tensor.to(torch.float32)

function_signature = {
    "name": "masked_select_fp32",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.bool)
    ],
    "outputs": [
        ((16,), torch.float32),
    ]
}
