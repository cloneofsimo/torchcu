
import torch

def sum_int8_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs element-wise multiplication and sum using int8.
    """
    input_int8 = input_tensor.to(torch.int8)
    weight_int8 = weight.to(torch.int8)
    output = (input_int8 * weight_int8).sum()
    return torch.tensor([output], dtype=torch.int32)

function_signature = {
    "name": "sum_int8_function",
    "inputs": [
        ((4, 4), torch.int8),
        ((4, 4), torch.int8)
    ],
    "outputs": [
        ((1,), torch.int32),
    ]
}
