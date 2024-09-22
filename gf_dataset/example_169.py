
import torch
import torch.nn.functional as F

def mish_int8_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Perform a simple linear transformation (matrix multiplication) and Mish activation,
    returning the result in int8. 
    """
    input_int8 = input_tensor.to(torch.int8)
    weight_int8 = weight.to(torch.int8)

    output = torch.matmul(input_int8, weight_int8.t())
    output = F.mish(output)
    output_int8 = torch.empty_like(output, dtype=torch.int8)
    torch.quantize_per_tensor(output, 0, 1, output_int8)

    return output_int8

function_signature = {
    "name": "mish_int8_function",
    "inputs": [
        ((4, 4), torch.int8),
        ((4, 4), torch.int8)
    ],
    "outputs": [
        ((4, 4), torch.int8),
    ]
}
