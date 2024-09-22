
import torch
import torch.nn.functional as F

def torch_celu_bf16_function(input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Applies the CELU activation function with specified alpha value.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    output = F.celu(input_bf16, alpha=alpha)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_celu_bfloat16_function",
    "inputs": [
        ((4, 4), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
