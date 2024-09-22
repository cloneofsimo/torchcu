
import torch

def wasserstein_bf16_loss(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes the Wasserstein loss between two input tensors using bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    target_bf16 = target_tensor.to(torch.bfloat16)
    loss = torch.mean(torch.abs(input_bf16 - target_bf16))
    return loss.to(torch.float32)

function_signature = {
    "name": "wasserstein_bf16_loss",
    "inputs": [
        ((1,), torch.float32),
        ((1,), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
