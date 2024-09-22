
import torch

def diagflat_int8_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a tensor to int8, then creates a diagonal matrix using torch.diagflat.
    """
    int8_tensor = input_tensor.to(torch.int8)
    output = torch.diagflat(int8_tensor)
    return output.to(torch.float32)

function_signature = {
    "name": "diagflat_int8_function",
    "inputs": [
        ((4,), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
