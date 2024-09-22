
import torch
import torch.nn.functional as F

def lightweight_attention_bf16(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                                 num_heads: int, dropout: float = 0.0) -> torch.Tensor:
    """
    Performs multi-head attention with bfloat16 precision for lightweight computation.

    Args:
        query: Query tensor (batch_size, seq_len, embedding_dim).
        key: Key tensor (batch_size, seq_len, embedding_dim).
        value: Value tensor (batch_size, seq_len, embedding_dim).
        num_heads: Number of attention heads.
        dropout: Dropout probability.

    Returns:
        Output tensor (batch_size, seq_len, embedding_dim).
    """
    batch_size, seq_len, embedding_dim = query.size()
    head_dim = embedding_dim // num_heads

    # Convert to bfloat16
    query = query.to(torch.bfloat16)
    key = key.to(torch.bfloat16)
    value = value.to(torch.bfloat16)

    # Reshape for multi-head attention
    query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # Calculate attention scores
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim**0.5)

    # Apply softmax and dropout
    attention_weights = F.softmax(attention_scores, dim=-1)
    attention_weights = F.dropout(attention_weights, p=dropout, training=self.training)

    # Weighted sum of values
    context = torch.matmul(attention_weights, value)

    # Concatenate heads and reshape
    context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embedding_dim)

    # Convert back to float32
    context = context.to(torch.float32)
    return context

function_signature = {
    "name": "lightweight_attention_bf16",
    "inputs": [
        ((10, 20, 512), torch.float32),
        ((10, 20, 512), torch.float32),
        ((10, 20, 512), torch.float32),
        (8, ), torch.int32
    ],
    "outputs": [
        ((10, 20, 512), torch.float32),
    ]
}
