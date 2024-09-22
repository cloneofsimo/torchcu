
import torch
import torch.nn.functional as F

def roberts_cross_gradient_loss_bf16(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Roberts Cross Gradient loss in bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    target_bf16 = target_tensor.to(torch.bfloat16)

    # Calculate gradients
    gx = input_bf16[:, 1:, :] - input_bf16[:, :-1, :]
    gy = input_bf16[:, :, 1:] - input_bf16[:, :, :-1]
    gtx = target_bf16[:, 1:, :] - target_bf16[:, :-1, :]
    gty = target_bf16[:, :, 1:] - target_bf16[:, :, :-1]

    # Calculate the loss
    loss = F.binary_cross_entropy(gx, gtx, reduction='none') + F.binary_cross_entropy(gy, gty, reduction='none')
    return loss.mean().to(torch.float32)

function_signature = {
    "name": "roberts_cross_gradient_loss_bf16",
    "inputs": [
        ((2, 4, 4), torch.float32),
        ((2, 4, 4), torch.float32)
    ],
    "outputs": [
        ((), torch.float32),
    ]
}
