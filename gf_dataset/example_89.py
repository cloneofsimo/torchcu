
import torch
from torch.nn import functional as F
from cutlass import *

class MyModule(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear(x)
        x = F.logspace(0, 1, steps=x.size(1), dtype=torch.float16) * x  # Apply logspace scaling
        return x

def torch_logspace_linear_fp16(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Applies a linear transformation with logspace scaling and fp16 precision.
    """
    model = MyModule(input_tensor.size(1), weight.size(0))
    model.linear.weight.data = weight
    model.linear.bias.data = bias
    return model(input_tensor).to(torch.float32)

function_signature = {
    "name": "torch_logspace_linear_fp16",
    "inputs": [
        ((16, 4), torch.float32),
        ((4, 8), torch.float32),
        ((8,), torch.float32)
    ],
    "outputs": [
        ((16, 8), torch.float32)
    ]
}
