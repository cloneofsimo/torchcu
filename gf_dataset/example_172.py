
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModule(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input_tensor):
        x = self.linear(input_tensor)
        x = F.avg_pool1d(x.unsqueeze(1), kernel_size=3, stride=2)  # Apply avg_pool1d
        x = torch.logspace(0, 1, steps=x.shape[1], dtype=torch.float32)  # Apply logspace
        return x

def torch_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Perform a series of operations on input tensor using bfloat16.
    """
    model = MyModule(input_tensor.shape[1], 10) 
    output = model(input_tensor.to(torch.float32))  # Forward pass with float32
    return output

function_signature = {
    "name": "torch_function",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 10), torch.float32),
    ]
}
