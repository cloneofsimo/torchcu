
import torch

def instance_norm_fp16_function(input_tensor: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Performs Instance Normalization with FP16 precision and returns the result in FP16.
    """
    input_tensor_fp16 = input_tensor.to(torch.float16)
    gamma_fp16 = gamma.to(torch.float16)
    beta_fp16 = beta.to(torch.float16)

    mean = torch.mean(input_tensor_fp16, dim=[2, 3], keepdim=True)
    variance = torch.var(input_tensor_fp16, dim=[2, 3], keepdim=True, unbiased=False)
    std = torch.sqrt(variance + 1e-5)  # Add small constant for numerical stability
    normalized_input = (input_tensor_fp16 - mean) / std
    output = (normalized_input * gamma_fp16) + beta_fp16
    return output.to(torch.float16)

function_signature = {
    "name": "instance_norm_fp16_function",
    "inputs": [
        ((4, 4, 8, 8), torch.float32),
        ((4,), torch.float32),
        ((4,), torch.float32),
    ],
    "outputs": [
        ((4, 4, 8, 8), torch.float16),
    ]
}
