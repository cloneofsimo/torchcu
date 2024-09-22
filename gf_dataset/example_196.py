
import torch
import torch.nn.functional as F
from torch.autograd import Function

class MyCustomFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor, weight, bias, lr):
        """
        Performs a linear transformation, applies ReLU activation, and updates the weight using SGD.

        Args:
            ctx: Context object for saving input tensors and hyperparameters.
            input_tensor: Input tensor of shape (batch_size, input_features).
            weight: Weight tensor of shape (output_features, input_features).
            bias: Bias tensor of shape (output_features,).
            lr: Learning rate for the SGD update.

        Returns:
            Output tensor of shape (batch_size, output_features).
        """
        ctx.save_for_backward(input_tensor, weight, bias)
        ctx.lr = lr

        output = F.linear(input_tensor, weight, bias)
        output = F.relu(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes gradients for the input tensor and weight.

        Args:
            ctx: Context object containing saved input tensors and hyperparameters.
            grad_output: Gradient of the loss w.r.t. the output tensor.

        Returns:
            Tuple of gradients for input tensor, weight, bias, and lr (None, as lr is not updated).
        """
        input_tensor, weight, bias = ctx.saved_tensors
        lr = ctx.lr

        # Compute gradients using chain rule
        grad_input = grad_output @ weight.t()
        grad_weight = grad_output.t() @ input_tensor
        grad_bias = grad_output.sum(dim=0)

        # Update weight using SGD
        weight -= lr * grad_weight

        return grad_input, grad_weight, grad_bias, None

def torch_custom_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, lr: float) -> torch.Tensor:
    """
    Calls the MyCustomFunction.apply() method to perform the forward and backward passes.

    Args:
        input_tensor: Input tensor of shape (batch_size, input_features).
        weight: Weight tensor of shape (output_features, input_features).
        bias: Bias tensor of shape (output_features,).
        lr: Learning rate for the SGD update.

    Returns:
        Output tensor of shape (batch_size, output_features).
    """
    return MyCustomFunction.apply(input_tensor, weight, bias, lr)

function_signature = {
    "name": "torch_custom_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32),
        (torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
