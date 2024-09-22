
import torch
import torch.nn as nn

class LogSumExpLoss(nn.Module):
    def __init__(self, dim=1):
        super(LogSumExpLoss, self).__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the LogSumExp loss, a common loss function for classification.
        
        Args:
            input (torch.Tensor): The input tensor, often representing scores or logits.
            target (torch.Tensor): The target labels, typically one-hot encoded.

        Returns:
            torch.Tensor: The computed LogSumExp loss.
        """
        # Calculate the log-sum-exp along the specified dimension
        lse = torch.logsumexp(input, dim=self.dim)
        # Calculate the cross-entropy loss
        loss = lse - torch.sum(input * target, dim=self.dim)
        return loss

def logsumexp_loss_function(input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute LogSumExp loss, a common loss function for classification.

    Args:
        input_tensor (torch.Tensor): The input tensor, often representing scores or logits.
        target (torch.Tensor): The target labels, typically one-hot encoded.

    Returns:
        torch.Tensor: The computed LogSumExp loss.
    """
    # Create an instance of the LogSumExpLoss class
    loss_fn = LogSumExpLoss()
    # Calculate the loss
    loss = loss_fn(input_tensor, target)
    return loss

function_signature = {
    "name": "logsumexp_loss_function",
    "inputs": [
        ((10, 5), torch.float32),
        ((10, 5), torch.float32)
    ],
    "outputs": [
        ((10,), torch.float32),
    ]
}
