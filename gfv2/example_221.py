
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MyModule, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.fc2(x)
        x = torch.log_softmax(x, dim=1)
        return x

def my_function(input_tensor: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Performs a classification task using a custom neural network.
    """
    model = MyModule(input_tensor.shape[1], labels.max() + 1)
    output = model(input_tensor.to(torch.bfloat16)).to(torch.float32)
    return output

function_signature = {
    "name": "my_function",
    "inputs": [
        ((1, 128), torch.float32),
        ((1,), torch.int64)
    ],
    "outputs": [
        ((1, 10), torch.float32)
    ]
}
