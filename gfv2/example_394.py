
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4, kernel_size=3, padding=1):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

def my_function(input_tensor: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, 
                  model: MyModel, alpha: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a grouped convolution, applies log softmax, calculates NLL loss, and
    adds L1 regularization on the weights. Returns the loss and the log probabilities.
    """
    # Move to fp16 for efficiency
    input_tensor = input_tensor.to(torch.float16)
    weight = weight.to(torch.float16)
    
    # Apply grouped convolution
    output = model(input_tensor)

    # Calculate log probabilities
    log_probs = F.log_softmax(output, dim=1)

    # Calculate NLL loss
    loss = F.nll_loss(log_probs, labels)

    # L1 regularization
    l1_reg = alpha * torch.sum(torch.abs(weight))
    loss += l1_reg

    return loss, log_probs

function_signature = {
    "name": "my_function",
    "inputs": [
        ((16, 3, 28, 28), torch.float32),  # Input tensor
        ((16, 3, 3, 3), torch.float32),   # Weight tensor
        ((16,), torch.int64),             # Labels
        (None, MyModel)                    # Model
    ],
    "outputs": [
        ((), torch.float32),                # Loss
        ((16, 10), torch.float32)           # Log probabilities
    ]
}

