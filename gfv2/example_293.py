
import torch

def addcmul_int8_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a simple addition with scaled multiplication using int8.
    """
    input_int8 = input_tensor.to(torch.int8)
    weight_int8 = weight.to(torch.int8)
    bias_int8 = bias.to(torch.int8)
    output = torch.addcmul(input_int8, 1.0, weight_int8, bias_int8)
    return output.to(torch.float32)

function_signature = {
    "name": "addcmul_int8_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
