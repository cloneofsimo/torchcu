
import torch
import torch.nn.functional as F

def my_loss_function(input_tensor: torch.Tensor, target_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Calculates a custom loss function.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    target_bf16 = target_tensor.to(torch.bfloat16)
    weights_bf16 = weights.to(torch.bfloat16)

    # Apply adaptive log softmax
    log_probs = F.adaptive_log_softmax(input_bf16, dim=1)

    # Calculate cross-entropy loss
    loss = F.nll_loss(log_probs, target_bf16, weight=weights_bf16, reduction='mean')

    # Apply smooth L1 loss on the input
    smooth_l1_loss = F.smooth_l1_loss(input_bf16, target_bf16, reduction='mean')

    # Combine losses with elementwise max
    combined_loss = torch.max(loss, smooth_l1_loss)

    return combined_loss.to(torch.float32)

function_signature = {
    "name": "my_loss_function",
    "inputs": [
        ((10, 5), torch.float32),
        ((10,), torch.long),
        ((5,), torch.float32),
    ],
    "outputs": [
        ((), torch.float32),
    ]
}
