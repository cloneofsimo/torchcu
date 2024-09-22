
import torch

def conv_tbc_int8_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Perform a simple convolution (with bias) using int8 data type.
    """
    input_int8 = input_tensor.to(torch.int8)
    weight_int8 = weight.to(torch.int8)
    bias_int8 = bias.to(torch.int8)
    output = torch.conv_tbc(input_int8, weight_int8, bias_int8, padding=1)
    return output.to(torch.float32)

function_signature = {
    "name": "conv_tbc_int8_function",
    "inputs": [
        ((1, 1, 4, 4), torch.float32),
        ((1, 1, 3, 3), torch.float32),
        ((1), torch.float32)
    ],
    "outputs": [
        ((1, 1, 4, 4), torch.float32),
    ]
}
