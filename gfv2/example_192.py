
import torch

def multihead_attention_fp32(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Performs multi-head attention with floating-point computations.

    Args:
        query: Query tensor of shape (batch_size, seq_len, hidden_dim).
        key: Key tensor of shape (batch_size, seq_len, hidden_dim).
        value: Value tensor of shape (batch_size, seq_len, hidden_dim).
        num_heads: Number of attention heads.

    Returns:
        Output tensor of shape (batch_size, seq_len, hidden_dim).
    """

    batch_size, seq_len, hidden_dim = query.size()

    # Split into heads
    query = query.view(batch_size, seq_len, num_heads, hidden_dim // num_heads).permute(0, 2, 1, 3)
    key = key.view(batch_size, seq_len, num_heads, hidden_dim // num_heads).permute(0, 2, 1, 3)
    value = value.view(batch_size, seq_len, num_heads, hidden_dim // num_heads).permute(0, 2, 1, 3)

    # Calculate attention scores
    attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (hidden_dim // num_heads)**0.5

    # Apply softmax to get attention weights
    attention_weights = torch.softmax(attention_scores, dim=-1)

    # Perform weighted sum of value
    output = torch.matmul(attention_weights, value)

    # Concatenate heads and reshape
    output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_dim)

    return output

function_signature = {
    "name": "multihead_attention_fp32",
    "inputs": [
        ((1, 16, 512), torch.float32),
        ((1, 16, 512), torch.float32),
        ((1, 16, 512), torch.float32),
        (8, torch.int32)
    ],
    "outputs": [
        ((1, 16, 512), torch.float32),
    ]
}
