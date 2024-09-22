
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by patch size."
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        # Patch Embedding
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

        # Class Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.randn(1, (image_size // patch_size) ** 2 + 1, dim))

        # Transformer Encoder
        self.transformer = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)]
        )

        # Head
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embed(x)  # (batch_size, dim, num_patches, num_patches)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, dim)

        # Class Token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # (batch_size, 1, dim)
        x = torch.cat([cls_token, x], dim=1)  # (batch_size, num_patches + 1, dim)

        # Positional Embedding
        x += self.pos_embed  # (batch_size, num_patches + 1, dim)

        # Transformer Encoder
        for block in self.transformer:
            x = block(x)

        # Classification Head
        cls_token = x[:, 0, :]  # (batch_size, dim)
        x = self.head(cls_token)  # (batch_size, num_classes)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(heads, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # Split into queries, keys, and values
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Reshape and transpose for multi-head attention
        q = q.view(x.shape[0], -1, self.heads, self.head_dim).transpose(1, 2)  # (batch_size, heads, num_patches + 1, head_dim)
        k = k.view(x.shape[0], -1, self.heads, self.head_dim).transpose(1, 2)  # (batch_size, heads, num_patches + 1, head_dim)
        v = v.view(x.shape[0], -1, self.heads, self.head_dim).transpose(1, 2)  # (batch_size, heads, num_patches + 1, head_dim)

        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch_size, heads, num_patches + 1, num_patches + 1)
        attn = torch.softmax(attn, dim=-1)  # (batch_size, heads, num_patches + 1, num_patches + 1)

        # Weighted sum of values
        out = (attn @ v).transpose(1, 2).contiguous().view(x.shape[0], -1, self.dim)  # (batch_size, num_patches + 1, dim)

        # Project output
        out = self.proj(out)

        return out

class MLP(nn.Module):
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Example usage
image_size = 224
patch_size = 16
num_classes = 1000
dim = 768
depth = 12
heads = 12
mlp_dim = 3072

model = VisionTransformer(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim)

# Input tensor
input_tensor = torch.randn(1, 3, image_size, image_size)

# Run the model
output = model(input_tensor)

# Print the output shape
print(output.shape)
