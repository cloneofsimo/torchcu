
import torch
import torch.nn as nn
import numpy as np

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.celu(x)
        x = self.fc2(x)
        return x

def my_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Loads a model, performs inference, and calculates Smooth L1 loss.
    """
    model = MyModel()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    output = model(input_tensor)
    loss = nn.SmoothL1Loss()(output, weight)
    return loss

function_signature = {
    "name": "my_function",
    "inputs": [
        ((10,), torch.float32),
        ((1,), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
