
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def torch_bilinear_int8_function(input1: torch.Tensor, input2: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a bilinear operation with int8 precision and ReLU activation.
    """
    torch.manual_seed(42)  # Ensure deterministic results
    input1_int8 = input1.to(torch.int8)
    input2_int8 = input2.to(torch.int8)
    weight_int8 = weight.to(torch.int8)
    bias_fp16 = bias.to(torch.float16)

    with autocast(enabled=True):
        output = F.linear(input1_int8, weight_int8, bias_fp16)
        output = F.linear(input2_int8, output, bias=None)  # Bilinear operation
        output = F.relu(output, inplace=True)  # ReLU activation, inplace for memory efficiency
    return output.to(torch.float32)

function_signature = {
    "name": "torch_bilinear_int8_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}

