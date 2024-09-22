
import torch

def addcmul_bf16_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a weighted sum of the input tensor and the weight tensor, adds a bias, and returns the result in bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)
    output = torch.addcmul(input_bf16, weight_bf16, bias_bf16, value=1.0)
    return output.to(torch.float32)

function_signature = {
    "name": "addcmul_bfloat16_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
