
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input_tensor):
        x = self.linear(input_tensor)
        x = torch.nn.functional.swish(x)
        x = torch.nn.functional.linear(x, torch.ones(x.shape[1]), bias=None)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        return x

function_signature = {
    "name": "my_module_forward",
    "inputs": [
        ((10,), torch.float32)
    ],
    "outputs": [
        ((10,), torch.float32),
    ]
}
