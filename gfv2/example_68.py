
import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def compute_nll_loss_with_rmse(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the negative log-likelihood (NLL) loss and the root mean squared error (RMSE).
    Applies a Scharr gradient to the input tensor before calculating the loss.
    
    Args:
      input_tensor: The predicted tensor.
      target_tensor: The ground truth tensor.

    Returns:
      A tuple containing the NLL loss and RMSE.
    """
    # Apply Scharr gradient
    input_tensor = torch.scharr(input_tensor, dim=1) 
    
    # Calculate NLL loss
    nll_loss = F.nll_loss(input_tensor, target_tensor)

    # Calculate RMSE
    rmse = torch.sqrt(torch.mean((input_tensor - target_tensor)**2))

    return nll_loss, rmse

function_signature = {
    "name": "compute_nll_loss_with_rmse",
    "inputs": [
        ((1, 1, 28, 28), torch.float32),
        ((1,), torch.long)
    ],
    "outputs": [
        ((), torch.float32),
        ((), torch.float32),
    ]
}
