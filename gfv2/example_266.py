
import torch

def conv_tbc_bf16_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Perform a simple 1D convolution with bias and ReLU activation using bfloat16.
    Input tensor is expected to be in Time-Batch-Channel (TBC) format.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)
    output = torch.nn.functional.conv1d(input_bf16, weight_bf16, bias_bf16, groups=1)
    return torch.relu(output).to(torch.float32)


function_signature = {
    "name": "conv_tbc_bf16_function",
    "inputs": [
        ((10, 1, 8), torch.float32),
        ((3, 8, 4), torch.float32),
        ((4,), torch.float32),
    ],
    "outputs": [
        ((10, 1, 4), torch.float32),
    ]
}
