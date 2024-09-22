
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModule(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = F.selu(x)
        x = F.adaptive_max_pool1d(x.unsqueeze(1), output_size=1).squeeze(1)
        return x

def my_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a linear transformation, SELU activation, adaptive max pooling, and element-wise division. 
    """
    x = torch.matmul(input_tensor, weight.t()) + bias
    x = F.selu(x)
    x = F.adaptive_max_pool1d(x.unsqueeze(1), output_size=1).squeeze(1)
    x.div_(2.0, out=x) # inplace division
    return x


function_signature = {
    "name": "my_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((4,), torch.float32),
    ]
}
