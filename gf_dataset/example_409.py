
import torch
import torch.nn.functional as F

def attention_module(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                    spatial_attention_weights: torch.Tensor, global_attention_weights: torch.Tensor,
                    spatial_attention_mask: torch.Tensor, global_attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Applies a multi-head attention mechanism with both spatial and global attention.

    Args:
        query: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        key: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        value: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        spatial_attention_weights: Spatial attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        global_attention_weights: Global attention weights of shape (batch_size, num_heads, seq_len, 1)
        spatial_attention_mask: Mask for spatial attention of shape (batch_size, num_heads, seq_len, seq_len)
        global_attention_mask: Mask for global attention of shape (batch_size, num_heads, seq_len, 1)

    Returns:
        Output tensor of shape (batch_size, seq_len, head_dim)
    """

    # Spatial Attention
    spatial_attention = torch.matmul(query, key.transpose(-1, -2))
    spatial_attention = spatial_attention + spatial_attention_weights
    spatial_attention = spatial_attention.masked_fill(spatial_attention_mask == 0, float('-inf'))
    spatial_attention = F.softmax(spatial_attention, dim=-1)

    # Global Attention
    global_attention = torch.matmul(query, key.transpose(-1, -2))
    global_attention = global_attention + global_attention_weights
    global_attention = global_attention.masked_fill(global_attention_mask == 0, float('-inf'))
    global_attention = F.softmax(global_attention, dim=-1)

    # Combine Spatial and Global Attention
    combined_attention = spatial_attention * spatial_attention_weights + global_attention * global_attention_weights

    # Weighted Sum of Values
    output = torch.matmul(combined_attention, value)

    return output.sum(dim=1)

function_signature = {
    "name": "attention_module",
    "inputs": [
        ((1, 4, 8, 16), torch.float32),
        ((1, 4, 8, 16), torch.float32),
        ((1, 4, 8, 16), torch.float32),
        ((1, 4, 8, 8), torch.float32),
        ((1, 4, 8, 1), torch.float32),
        ((1, 4, 8, 8), torch.int8),
        ((1, 4, 8, 1), torch.int8)
    ],
    "outputs": [
        ((1, 8, 16), torch.float32)
    ]
}
