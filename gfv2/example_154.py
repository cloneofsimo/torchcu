
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = nn.functional.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

def se_subtract_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Applies SEBlock, subtracts weight, and returns the result in fp32.
    """
    se_block = SEBlock(input_tensor.shape[1])
    se_output = se_block(input_tensor.to(torch.bfloat16))
    se_output = se_output.to(torch.float32)
    output = se_output - weight.to(torch.float32)
    return output

function_signature = {
    "name": "se_subtract_function",
    "inputs": [
        ((1, 128, 28, 28), torch.float32),
        ((128,), torch.float32)
    ],
    "outputs": [
        ((1, 128, 28, 28), torch.float32),
    ]
}
