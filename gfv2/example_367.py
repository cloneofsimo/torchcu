
import torch
import torch.nn as nn

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.window_size = window_size

    def forward(self, x):
        # Slice input into windows
        B, H, W, C = x.shape
        x = x.reshape(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * (H // self.window_size) * (W // self.window_size), self.window_size * self.window_size, C)

        # Attention module
        x = self.norm1(x)
        x = x.permute(1, 0, 2)  # (N, B, C)
        x, _ = self.attn(x, x, x)  # (N, B, C)
        x = x.permute(1, 0, 2)  # (B, N, C)

        # MLP module
        x = self.norm2(x)
        x = self.mlp(x)

        # Reshape back to original shape
        x = x.reshape(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        return x

def my_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on a 4D input tensor:

    1. Slices the input tensor along the first dimension.
    2. Performs matrix multiplication with the given weight.
    3. Applies a Swin Transformer block.
    4. Performs an in-place addition with the input tensor.
    5. Returns the result.
    """
    # Slice input tensor
    sliced_input = input_tensor[0:2, :, :, :]

    # Matrix multiplication
    output = torch.matmul(sliced_input.reshape(sliced_input.shape[0], -1), weight)

    # Swin Transformer block
    output = output.reshape(sliced_input.shape[0], sliced_input.shape[1], sliced_input.shape[2], sliced_input.shape[3])
    output = SwinTransformerBlock(dim=output.shape[-1], num_heads=4, window_size=2)(output)

    # In-place addition
    input_tensor[0:2, :, :, :] += output

    return input_tensor

function_signature = {
    "name": "my_function",
    "inputs": [
        ((4, 4, 4, 4), torch.float32),
        ((16, 16), torch.float32),
    ],
    "outputs": [
        ((4, 4, 4, 4), torch.float32),
    ]
}
