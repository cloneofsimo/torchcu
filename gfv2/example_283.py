
import torch
import torch.nn.functional as F

def bce_swish_max_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on input tensor with bfloat16 precision for efficiency:
    1. Applies a linear transformation with provided weight and bias.
    2. Applies binary cross entropy with sigmoid activation.
    3. Applies Swish activation (x * sigmoid(x)).
    4. Takes the maximum value along the last dimension.
    
    Returns a tensor with the maximum values.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    output = torch.matmul(input_bf16, weight_bf16.t()) + bias_bf16
    output = F.binary_cross_entropy_with_logits(output, torch.ones_like(output), reduction='none')
    output = output * torch.sigmoid(output)  # Swish activation
    output = torch.max(output, dim=-1).values  # Maximum along last dimension
    return output.to(torch.float32)

function_signature = {
    "name": "bce_swish_max_function",
    "inputs": [
        ((1, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32),
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
