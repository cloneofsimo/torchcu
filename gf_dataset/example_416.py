
import torch

def torch_softmax_diagflat_squeeze(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies softmax to the input tensor, then extracts the diagonal elements and squeezes the result.
    """
    softmax_output = torch.softmax(input_tensor, dim=1)
    diag_output = torch.diagflat(softmax_output)
    return torch.squeeze(diag_output)

function_signature = {
    "name": "torch_softmax_diagflat_squeeze",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4,), torch.float32),
    ]
}
