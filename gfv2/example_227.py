
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gradient_penalty_function(input_tensor: torch.Tensor, output_tensor: torch.Tensor, 
                              weight: torch.Tensor, lambda_: float) -> torch.Tensor:
    """
    Computes the gradient penalty for the Wasserstein-1 GAN loss.
    """
    # Interpolate between real and fake samples
    alpha = torch.rand(input_tensor.size(0), 1, 1, 1).to(input_tensor.device)
    interpolated_tensor = alpha * input_tensor + (1 - alpha) * output_tensor

    # Calculate the gradient of the discriminator w.r.t. the interpolated samples
    interpolated_tensor.requires_grad = True
    interpolated_output = nn.functional.adaptive_avg_pool1d(F.softshrink(interpolated_tensor), 1)
    interpolated_output = interpolated_output.view(interpolated_output.size(0), -1)
    interpolated_output = torch.matmul(interpolated_output, weight)

    # Calculate the gradient norm
    grad_output = torch.ones_like(interpolated_output)
    grad_input = torch.autograd.grad(
        outputs=interpolated_output,
        inputs=interpolated_tensor,
        grad_outputs=grad_output,
        create_graph=True,
        retain_graph=True,
    )[0].view(interpolated_tensor.size(0), -1)
    grad_norm = grad_input.norm(2, 1)

    # Calculate the gradient penalty
    grad_penalty = lambda_ * ((grad_norm - 1) ** 2).mean()

    return grad_penalty

function_signature = {
    "name": "gradient_penalty_function",
    "inputs": [
        ((10, 1, 28, 28), torch.float32),
        ((10, 1, 28, 28), torch.float32),
        ((784, 1), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
