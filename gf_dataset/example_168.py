
import torch
import torch.nn.functional as F

def torch_feature_mixing_prelu_fp32_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
    """
    Performs feature mixing, PReLU activation, and returns the result in fp32.
    """
    output = F.linear(input_tensor, weight, bias)
    output = F.prelu(output, slope)
    return output

function_signature = {
    "name": "torch_feature_mixing_prelu_fp32_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
