
import torch

def sigmoid_pow_expand_bf16_function(input_tensor: torch.Tensor, exponent: float) -> torch.Tensor:
    """
    Applies sigmoid, raises to a power, and expands the result.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    sigmoid_output = torch.sigmoid(input_bf16)
    powered_output = torch.pow(sigmoid_output, exponent)
    expanded_output = powered_output.expand(input_tensor.size(0), -1, 5)
    return expanded_output.to(torch.float32)

function_signature = {
    "name": "sigmoid_pow_expand_bf16_function",
    "inputs": [
        ((1, 10), torch.float32),
        (torch.float32,)
    ],
    "outputs": [
        ((1, 10, 5), torch.float32),
    ]
}
