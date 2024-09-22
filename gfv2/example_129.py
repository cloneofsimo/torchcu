
import torch

def spectral_contrast_inplace_bf16(input_tensor: torch.Tensor, scale: float = 10.0, bias: float = 1.0) -> torch.Tensor:
    """
    Applies spectral contrast normalization to the input tensor inplace, using bfloat16 for efficiency.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    mean = input_bf16.mean(dim=1, keepdim=True)
    std = input_bf16.std(dim=1, keepdim=True)
    input_bf16.sub_(mean)
    input_bf16.div_(std)
    input_bf16.mul_(scale)
    input_bf16.add_(bias)
    input_bf16.abs_()
    return input_tensor.to(torch.float32)


function_signature = {
    "name": "spectral_contrast_inplace_bf16",
    "inputs": [
        ((4, 10), torch.float32), 
        ((1,), torch.float32), 
        ((1,), torch.float32)
    ],
    "outputs": [
        ((4, 10), torch.float32)
    ]
}
