
import torch

def complex_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a transposed 3D convolution with bfloat16 precision, applies ReLU activation, 
    and then calculates the minimum value along a specified dimension. 
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)
    output = torch.nn.functional.conv_transpose3d(input_bf16, weight_bf16, bias_bf16, stride=2, padding=1, output_padding=1)
    output = torch.relu(output).to(torch.float32)
    min_values = torch.min(output, dim=1, keepdim=False)[0]
    return min_values

function_signature = {
    "name": "complex_function",
    "inputs": [
        ((16, 16, 4, 4, 4), torch.float32),
        ((8, 8, 4, 4, 4), torch.float32),
        ((8,), torch.float32)
    ],
    "outputs": [
        ((16, 4, 4, 4), torch.float32)
    ]
}
