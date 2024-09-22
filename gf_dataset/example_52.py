
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyReshapeLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyReshapeLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input_tensor):
        # Reshape the input tensor
        batch_size = input_tensor.size(0)
        input_tensor = input_tensor.view(batch_size, -1)
        
        # Linear transformation
        output = self.linear(input_tensor)
        return output

def torch_reshape_linear_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a reshape and linear transformation.
    """
    model = MyReshapeLinear(in_features=16, out_features=8)
    return model(input_tensor)

function_signature = {
    "name": "torch_reshape_linear_function",
    "inputs": [
        ((2, 4, 4), torch.float32),
    ],
    "outputs": [
        ((2, 8), torch.float32)
    ]
}
