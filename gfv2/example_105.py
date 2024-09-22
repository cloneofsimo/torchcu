
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.fc = nn.Linear(hidden_channels, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 3, 4, 1)  # N, H, W, D, C -> N, C, H, W, D
        x = self.encoder(x)
        x = self.pool(x).squeeze()
        x = self.fc(x)
        return x

def my_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies a 3D convolutional layer, followed by a transformer encoder, 
    adaptive max pooling, and a fully connected layer. The output is a tensor 
    of shape (batch_size, 10).
    """
    model = MyModule(in_channels=3, hidden_channels=64, num_heads=4)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor.int())
    return output

function_signature = {
    "name": "my_function",
    "inputs": [
        ((16, 3, 32, 32, 32), torch.int8)
    ],
    "outputs": [
        ((16, 10), torch.float32)
    ]
}
