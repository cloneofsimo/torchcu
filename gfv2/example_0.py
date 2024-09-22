
import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalRegularization(nn.Module):
    def __init__(self, weight, weight_decay=0.001):
        super().__init__()
        self.weight = weight
        self.weight_decay = weight_decay

    def forward(self, input):
        # Calculate orthogonal regularization loss
        W = self.weight
        I = torch.eye(W.shape[0], dtype=torch.float32, device=W.device)
        loss = self.weight_decay * (torch.norm(torch.matmul(W.T, W) - I) ** 2)
        return loss

class MyModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.orthogonal_reg = OrthogonalRegularization(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        orthogonal_loss = self.orthogonal_reg(None)  # Apply orthogonal regularization
        return x, orthogonal_loss  # Return output and orthogonal loss

def my_function(input_tensor: torch.Tensor, size: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Performs a grid sampling operation with orthogonal regularization on the weights.
    """

    # 1. Generate grid using affine_grid
    grid = F.affine_grid(size, input_tensor.size())

    # 2. Sample using grid_sample
    output = F.grid_sample(input_tensor, grid, align_corners=True)

    # 3. Apply orthogonal regularization on the weights
    model = MyModule(input_tensor.shape[1], weights.shape[1])
    output, orthogonal_loss = model(output.to(torch.float16))

    # 4. Return the output
    return output.to(torch.float32), orthogonal_loss

function_signature = {
    "name": "my_function",
    "inputs": [
        ((2, 4, 4, 4), torch.float32),  # Input Tensor (batch, channels, height, width)
        ((2, 3, 2), torch.float32),  # Size tensor (batch, output height, output width)
        ((4, 4), torch.float32)  # Weights tensor (input channels, output channels)
    ],
    "outputs": [
        ((2, 4, 2, 2), torch.float32), # Output Tensor (batch, channels, height, width)
        ((), torch.float32),          # Orthogonal Loss
    ]
}
