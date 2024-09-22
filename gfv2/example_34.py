
import torch
import torch.nn.functional as F
from torch.autograd import Function

class ImageJacobian(Function):
    @staticmethod
    def forward(ctx, input_tensor, weight, bias):
        """
        Compute the Jacobian of the image with respect to the input tensor.
        """
        ctx.save_for_backward(input_tensor, weight, bias)
        output = F.conv2d(input_tensor, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the gradients of the input tensor, weight and bias.
        """
        input_tensor, weight, bias = ctx.saved_tensors

        # Compute the Jacobian of the image with respect to the input tensor
        grad_input = torch.autograd.grad(
            outputs=output, 
            inputs=input_tensor, 
            grad_outputs=grad_output, 
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute the gradients of the weight and bias
        grad_weight = torch.autograd.grad(
            outputs=output, 
            inputs=weight, 
            grad_outputs=grad_output, 
            create_graph=True,
            retain_graph=True
        )[0]
        grad_bias = torch.autograd.grad(
            outputs=output, 
            inputs=bias, 
            grad_outputs=grad_output, 
            create_graph=True,
            retain_graph=True
        )[0]
        
        return grad_input, grad_weight, grad_bias

def image_jacobian_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Compute the Jacobian of the image with respect to the input tensor.
    """
    return ImageJacobian.apply(input_tensor, weight, bias)

function_signature = {
    "name": "image_jacobian_function",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),
        ((3, 3, 3, 3), torch.float32),
        ((3,), torch.float32)
    ],
    "outputs": [
        ((1, 3, 224, 224), torch.float32),
    ]
}
