
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, in_features, out_features, dropout_p=0.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(p=dropout_p)
        self.rrelu = nn.RReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = self.rrelu(x)
        return x

def torch_module_forward(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs forward pass through a simple module with linear, dropout, and RReLU layers.
    """
    model = MyModule(in_features=4, out_features=8)
    return model(input_tensor)

function_signature = {
    "name": "torch_module_forward",
    "inputs": [
        ((4,), torch.float32),
    ],
    "outputs": [
        ((8,), torch.float32),
    ]
}

