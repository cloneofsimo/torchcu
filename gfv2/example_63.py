
import torch

def glu_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Applies the Gated Linear Units (GLU) activation function.
    """
    input_fp16 = input_tensor.to(torch.float16)
    weight_fp16 = weight.to(torch.float16)
    bias_fp16 = bias.to(torch.float16)
    
    linear_output = torch.matmul(input_fp16, weight_fp16) + bias_fp16
    
    linear_output_a, linear_output_b = torch.split(linear_output, linear_output.shape[1] // 2, dim=1)
    
    output = linear_output_a * torch.sigmoid(linear_output_b)
    
    return output.to(torch.float32)

function_signature = {
    "name": "glu_function",
    "inputs": [
        ((4, 8), torch.float32),
        ((8, 8), torch.float32),
        ((8,), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
