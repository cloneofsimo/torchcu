
import torch
import torch.nn.functional as F

def fused_linear_int8_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Perform a fused linear transformation (matrix multiplication + bias) and ReLU activation using int8.
    """
    input_int8 = input_tensor.to(torch.int8)
    weight_int8 = weight.to(torch.int8)
    bias_int8 = bias.to(torch.int8)
    output = F.linear(input_int8, weight_int8, bias_int8)
    return F.relu(output).to(torch.float32)

function_signature = {
    "name": "fused_linear_int8_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
