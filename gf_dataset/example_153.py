
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModule(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input_tensor):
        x = self.linear(input_tensor)
        x = torch.exp(x)
        x = torch.div(x, input_tensor, rounding_mode='trunc')  # elementwise_div
        return x

def torch_module_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies a custom module with linear, exp, and elementwise_div operations.
    """
    module = MyModule(in_features=4, out_features=4)
    output = module(input_tensor)
    return output

function_signature = {
    "name": "torch_module_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
