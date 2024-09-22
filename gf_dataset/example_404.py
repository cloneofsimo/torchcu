
import torch
import torch.nn.functional as F

def torch_fused_dropout_linear_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, p: float) -> torch.Tensor:
    """
    Performs fused dropout, linear transformation, and ReLU activation.
    """
    output = F.dropout(input_tensor, p=p, training=True, inplace=True)
    output = F.linear(output, weight, bias)
    output = F.relu(output)
    return output

function_signature = {
    "name": "torch_fused_dropout_linear_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32),
        (0.5, None)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}

