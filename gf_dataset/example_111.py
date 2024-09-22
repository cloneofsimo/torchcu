
import torch

def flash_attention_fp16(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Performs Flash Attention with fp16 precision.

    Args:
        query: Query tensor of shape (batch, head, seq_len, head_dim)
        key: Key tensor of shape (batch, head, seq_len, head_dim)
        value: Value tensor of shape (batch, head, seq_len, head_dim)
        mask: Optional mask tensor of shape (batch, seq_len) 

    Returns:
        Output tensor of shape (batch, head, seq_len, head_dim)
    """
    query = query.to(torch.float16)
    key = key.to(torch.float16)
    value = value.to(torch.float16)

    # Calculate attention scores
    attention_scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / (key.shape[-1] ** 0.5)

    # Apply mask if provided
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

    # Calculate softmax probabilities
    attention_probs = torch.softmax(attention_scores, dim=-1).to(torch.float16)

    # Perform weighted sum of values
    output = torch.einsum('bhqk,bhkd->bhqd', attention_probs, value).to(torch.float32)
    return output

function_signature = {
    "name": "flash_attention_fp16",
    "inputs": [
        ((4, 8, 32, 64), torch.float32),
        ((4, 8, 32, 64), torch.float32),
        ((4, 8, 32, 64), torch.float32),
        ((4, 32), torch.bool)
    ],
    "outputs": [
        ((4, 8, 32, 64), torch.float32),
    ]
}
