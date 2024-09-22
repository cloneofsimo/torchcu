
import torch

def torch_group_norm_int8_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, num_groups: int) -> torch.Tensor:
    """
    Performs group normalization with int8 output.
    """
    input_tensor = input_tensor.to(torch.float32)
    weight = weight.to(torch.float32)
    bias = bias.to(torch.float32)

    output = torch.nn.functional.group_norm(input_tensor, num_groups=num_groups, weight=weight, bias=bias)
    output = output.to(torch.int8)
    return output

function_signature = {
    "name": "torch_group_norm_int8_function",
    "inputs": [
        ((2, 3, 4, 4), torch.float32),  # Input tensor
        ((3,), torch.float32),          # Weight tensor
        ((3,), torch.float32),          # Bias tensor
        (3, torch.int32)                # Number of groups
    ],
    "outputs": [
        ((2, 3, 4, 4), torch.int8),      # Output tensor
    ]
}
