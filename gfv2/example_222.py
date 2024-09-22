
import torch
import torch.nn.functional as F

def multi_loss_function(input_tensor: torch.Tensor, target_tensor: torch.Tensor, center_weights: torch.Tensor, 
                        alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.1) -> list:
    """
    Calculates a combined loss function with Smooth L1, RReLU, Poisson, and Center loss components.

    Args:
        input_tensor (torch.Tensor): Input tensor of size [batch_size, ...]
        target_tensor (torch.Tensor): Target tensor of size [batch_size, ...]
        center_weights (torch.Tensor): Weights for the center loss, size [batch_size, num_centers]
        alpha (float, optional): Weight for Smooth L1 loss. Defaults to 1.0.
        beta (float, optional): Weight for RReLU loss. Defaults to 0.5.
        gamma (float, optional): Weight for Poisson loss. Defaults to 0.1.

    Returns:
        list: A list containing the following losses:
            - Smooth L1 loss
            - RReLU loss
            - Poisson loss
            - Center loss
    """
    # Smooth L1 Loss
    smooth_l1_loss = F.smooth_l1_loss(input_tensor, target_tensor, reduction='mean')

    # RReLU Loss
    rrelu_loss = F.rrelu(input_tensor, lower=0.1, upper=0.3, training=True)
    rrelu_loss = torch.mean(torch.abs(rrelu_loss - target_tensor))

    # Poisson Loss
    poisson_loss = F.poisson_nll_loss(input_tensor, target_tensor, log_input=False, full=False, reduction='mean')

    # Center Loss
    center_loss = torch.sum((input_tensor - center_weights)**2, dim=1)
    center_loss = torch.mean(center_loss)

    # Combine Losses
    total_loss = alpha * smooth_l1_loss + beta * rrelu_loss + gamma * poisson_loss + center_loss

    return [smooth_l1_loss, rrelu_loss, poisson_loss, center_loss, total_loss]

function_signature = {
    "name": "multi_loss_function",
    "inputs": [
        ((1,), torch.float32),
        ((1,), torch.float32),
        ((1, 10), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
        ((1,), torch.float32),
        ((1,), torch.float32),
        ((1,), torch.float32),
        ((1,), torch.float32)
    ]
}
