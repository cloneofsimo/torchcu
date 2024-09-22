
import torch
import torch.nn as nn
from torch.nn import functional as F

class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_layers=12, num_heads=12, hidden_size=768, mlp_dim=3072, num_classes=1000):
        super(VisionTransformer, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes

        # Patch embedding
        self.patch_embedding = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, (image_size // patch_size) ** 2 + 1, hidden_size))

        # Transformer encoder
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads, mlp_dim) for _ in range(num_layers)
        ])

        # Classification head
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embedding(x)  # (batch_size, hidden_size, seq_len, seq_len)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, seq_len, hidden_size)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # (batch_size, 1, hidden_size)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, 1 + seq_len, hidden_size)
        x += self.pos_embedding  # (batch_size, 1 + seq_len, hidden_size)

        # Transformer encoder
        for encoder_layer in self.transformer_encoder:
            x = encoder_layer(x)

        # Classification head
        cls_token = x[:, 0, :]  # (batch_size, hidden_size)
        output = self.head(cls_token)  # (batch_size, num_classes)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Multi-head attention
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output

        # MLP
        x = self.norm2(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        return x

def vit_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Vision Transformer function for image classification.
    """
    model = VisionTransformer()
    model.eval()
    output = model(input_tensor)
    return output

function_signature = {
    "name": "vit_function",
    "inputs": [
        ((3, 224, 224), torch.float32),
    ],
    "outputs": [
        ((1000,), torch.float32),
    ]
}
