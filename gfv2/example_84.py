
import torch

def instance_norm_bf16_function(input_tensor: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Performs instance normalization on the input tensor using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    gamma_bf16 = gamma.to(torch.bfloat16)
    beta_bf16 = beta.to(torch.bfloat16)

    mean = input_bf16.mean(dim=1, keepdim=True)
    var = input_bf16.var(dim=1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps).to(torch.bfloat16)
    output = (input_bf16 - mean) / std
    output = output * gamma_bf16 + beta_bf16
    return output.to(torch.float32)

function_signature = {
    "name": "instance_norm_bfloat16_function",
    "inputs": [
        ((4, 4, 4), torch.float32),
        ((4,), torch.float32),
        ((4,), torch.float32),
        (float,)
    ],
    "outputs": [
        ((4, 4, 4), torch.float32),
    ]
}
