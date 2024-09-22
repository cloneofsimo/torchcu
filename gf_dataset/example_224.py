
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd

class MyCustomModule(torch.autograd.Function):
    @staticmethod
    @custom_bwd
    def forward(ctx, input_tensor, weight):
        ctx.save_for_backward(input_tensor, weight)
        output = F.relu(F.linear(input_tensor, weight))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight = ctx.saved_tensors
        grad_input = grad_output @ weight.t()
        grad_weight = (input_tensor.t() @ grad_output).t()
        return grad_input, grad_weight

def torch_custom_module(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Perform a simple linear transformation and ReLU activation.
    """
    return MyCustomModule.apply(input_tensor, weight)

function_signature = {
    "name": "torch_custom_module",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
