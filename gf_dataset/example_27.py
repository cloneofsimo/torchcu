
import torch

def torch_mse_loss_bf16_function(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Mean Squared Error (MSE) loss between two tensors using bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    target_bf16 = target_tensor.to(torch.bfloat16)
    diff = input_bf16 - target_bf16
    squared_diff = diff * diff
    loss = squared_diff.sum() / squared_diff.numel()
    return loss.to(torch.float32)

function_signature = {
    "name": "torch_mse_loss_bf16_function",
    "inputs": [
        ((10,), torch.float32),
        ((10,), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
