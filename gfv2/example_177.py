
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrunedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, sparsity_ratio=0.5):
        super(PrunedLinear, self).__init__(in_features, out_features, bias=bias)
        self.sparsity_ratio = sparsity_ratio

    def forward(self, input):
        # Apply pruning (simplistic example, real pruning is more complex)
        mask = torch.rand(self.weight.shape) > self.sparsity_ratio
        pruned_weight = self.weight * mask
        output = F.linear(input, pruned_weight, self.bias)
        return output

class MyModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MyModel, self).__init__()
        self.fc1 = PrunedLinear(in_features, hidden_features, sparsity_ratio=0.5)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def my_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Simple model with pruning, auto-mixed precision, and fused dropout.
    """
    model = MyModel(in_features=10, hidden_features=5, out_features=2)
    model.to(torch.float16)
    model.fc1.weight.data = model.fc1.weight.data.to(torch.float16)
    model.fc1.bias.data = model.fc1.bias.data.to(torch.float16)
    model.fc2.weight.data = model.fc2.weight.data.to(torch.float16)
    model.fc2.bias.data = model.fc2.bias.data.to(torch.float16)
    model.train()
    with torch.cuda.amp.autocast():
        output = model(input_tensor.to(torch.float16))
    return output.to(torch.float32)


function_signature = {
    "name": "my_function",
    "inputs": [
        ((10,), torch.float32),
    ],
    "outputs": [
        ((2,), torch.float32),
    ]
}

