
import torch

def complex_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a series of operations on the input tensor, including:
        - Matrix multiplication with weight
        - Addition of bias
        - SELU activation
        - All-reduce operation across all devices
        - Output tensor is converted to bfloat16
        - A second tensor is returned with its values multiplied by 2
    """
    output = torch.matmul(input_tensor, weight.t())
    output = output + bias
    output = torch.selu(output)
    output = torch.distributed.all_reduce(output)
    output_bf16 = output.to(torch.bfloat16)
    output2 = 2 * input_tensor
    return output_bf16, output2

function_signature = {
    "name": "complex_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.bfloat16),
        ((4, 4), torch.float32),
    ]
}
