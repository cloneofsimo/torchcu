
import torch
from torch import nn
from torch.cuda.amp import autocast

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls'):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.pool = pool

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim), num_layers=depth)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        x = x[:, 0] if self.pool == 'cls' else x.mean(dim=1)
        return self.mlp_head(x)

def vision_transformer_fp32_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of a Vision Transformer model with gradient checkpointing.
    """
    with torch.cuda.amp.autocast(enabled=False):
        model = VisionTransformer(image_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072)
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        return output

function_signature = {
    "name": "vision_transformer_fp32_function",
    "inputs": [
        ((3, 224, 224), torch.float32),
    ],
    "outputs": [
        ((1000), torch.float32),
    ]
}
