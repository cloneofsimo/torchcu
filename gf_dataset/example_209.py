
import torch
import torch.nn.functional as F
from torch.autograd import Function

class SoftmaxTemperatureFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Applies softmax with temperature scaling.
        """
        ctx.save_for_backward(input_tensor, torch.tensor(temperature))
        return F.softmax(input_tensor / temperature, dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Computes the backward pass.
        """
        input_tensor, temperature = ctx.saved_tensors
        grad_input = grad_output * (input_tensor / temperature - F.softmax(input_tensor / temperature, dim=-1))
        return grad_input, None

def torch_softmax_temperature_function(input_tensor: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Wrapper function for the custom SoftmaxTemperatureFunction.
    """
    return SoftmaxTemperatureFunction.apply(input_tensor, temperature)

function_signature = {
    "name": "torch_softmax_temperature_function",
    "inputs": [
        ((16, 16, 16, 3), torch.float32),
        (torch.float32,)
    ],
    "outputs": [
        ((16, 16, 16, 3), torch.float32)
    ]
}
