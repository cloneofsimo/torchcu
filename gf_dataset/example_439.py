
import torch

def torch_conv1d_bf16_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a 1D convolution with bfloat16 precision and returns the result in float32.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)
    output_bf16 = torch.nn.functional.conv1d(input_bf16, weight_bf16, bias_bf16)
    return output_bf16.to(torch.float32)

function_signature = {
    "name": "torch_conv1d_bfloat16_function",
    "inputs": [
        ((1, 16, 100), torch.float32),
        ((3, 16, 5), torch.float32),
        ((3,), torch.float32)
    ],
    "outputs": [
        ((1, 3, 98), torch.float32)
    ]
}
