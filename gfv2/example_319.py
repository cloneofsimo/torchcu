
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch import Tensor
import numpy as np

class RobustLossFunction(Function):
    @staticmethod
    def forward(ctx, input: Tensor, target: Tensor, alpha: float = 1.0):
        """
        Forward pass of the robust loss function.

        Args:
            ctx: Context object used to store intermediate values for backward pass.
            input: Input tensor.
            target: Target tensor.
            alpha: Hyperparameter controlling robustness (default: 1.0).

        Returns:
            Loss tensor.
        """
        ctx.save_for_backward(input, target)
        ctx.alpha = alpha
        
        return F.mse_loss(input, target)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """
        Backward pass of the robust loss function.

        Args:
            ctx: Context object containing saved tensors.
            grad_output: Gradient of the loss with respect to the output.

        Returns:
            Gradients of the loss with respect to the input and target.
        """
        input, target = ctx.saved_tensors
        alpha = ctx.alpha

        # Calculate gradients for input and target
        grad_input = grad_output * (input - target)
        grad_target = -grad_output * (input - target)

        # Apply robustness scaling factor
        grad_input = grad_input * (1.0 - alpha * torch.sign(grad_input))
        grad_target = grad_target * (1.0 - alpha * torch.sign(grad_target))

        return grad_input, grad_target, None

def robust_loss(input: Tensor, target: Tensor, alpha: float = 1.0) -> Tensor:
    """
    Computes the robust loss.

    Args:
        input: Input tensor.
        target: Target tensor.
        alpha: Hyperparameter controlling robustness (default: 1.0).

    Returns:
        Loss tensor.
    """
    return RobustLossFunction.apply(input, target, alpha)

def gumbel_softmax(logits: Tensor, tau: float = 1.0, hard: bool = False, dim: int = -1) -> Tensor:
    """
    Samples from a Gumbel-Softmax distribution.

    Args:
        logits: Unnormalized log probabilities.
        tau: Temperature parameter (default: 1.0).
        hard: If True, returns a one-hot vector (default: False).
        dim: Dimension over which to apply the Gumbel-Softmax (default: -1).

    Returns:
        Sampled probabilities.
    """
    y_soft = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=dim)
    return y_soft

def forward(x: Tensor, weights: Tensor, bias: Tensor) -> Tensor:
    """
    Forward pass of a simple linear layer with ReLU activation.

    Args:
        x: Input tensor.
        weights: Weight tensor.
        bias: Bias tensor.

    Returns:
        Output tensor.
    """
    return F.relu(torch.matmul(x, weights.t()) + bias)


function_signature = {
    "name": "robust_loss_forward_gumbel_softmax_linear",
    "inputs": [
        ((10,), torch.float16),
        ((10,), torch.float16),
        (1.0, torch.float32),
        ((10, 10), torch.float16),
        ((10,), torch.float16)
    ],
    "outputs": [
        ((10,), torch.float16),
        ((10,), torch.float16)
    ]
}
