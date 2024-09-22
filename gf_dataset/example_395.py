
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import ifftshift

class PruningLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, pruning_ratio=0.5):
        super().__init__(in_features, out_features, bias=bias)
        self.pruning_ratio = pruning_ratio

    def forward(self, x):
        # Apply pruning
        mask = torch.rand(self.weight.shape) > self.pruning_ratio
        self.weight.data *= mask.float()
        return F.linear(x, self.weight, self.bias)

class ContrastiveModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.linear1 = PruningLinear(in_features, hidden_features, pruning_ratio=0.2)
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x1, x2):
        # Forward pass for both inputs
        h1 = F.relu(self.linear1(x1))
        h2 = F.relu(self.linear1(x2))
        
        # Calculate contrastive loss
        similarity = F.cosine_similarity(h1, h2)
        loss = 1 - similarity
        
        return loss

def torch_contrastive_loss_ifftshift(input1: torch.Tensor, input2: torch.Tensor, model: ContrastiveModel) -> torch.Tensor:
    """
    Calculates the contrastive loss between two inputs using a pruned linear model,
    with ifftshift applied to the output of the first linear layer.
    """
    loss = model(input1, input2)
    return loss

function_signature = {
    "name": "torch_contrastive_loss_ifftshift",
    "inputs": [
        ((16, 16), torch.float32),
        ((16, 16), torch.float32),
        None,
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
