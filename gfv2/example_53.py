
import torch
import torch.nn.functional as F

def causal_attention_bf16(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    Applies causal self-attention with bfloat16 precision.
    
    Args:
        query: (batch_size, seq_len, hidden_dim)
        key: (batch_size, seq_len, hidden_dim)
        value: (batch_size, seq_len, hidden_dim)

    Returns:
        output: (batch_size, seq_len, hidden_dim)
    """
    
    # Convert to bfloat16
    query_bf16 = query.to(torch.bfloat16)
    key_bf16 = key.to(torch.bfloat16)
    value_bf16 = value.to(torch.bfloat16)
    
    # Calculate attention scores
    scores = torch.bmm(query_bf16, key_bf16.transpose(1, 2)) / (query_bf16.shape[-1] ** 0.5)
    
    # Apply causal mask
    mask = torch.triu(torch.ones(scores.shape[-2], scores.shape[-1]), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    
    # Softmax normalization
    attention_weights = F.softmax(scores, dim=-1)
    
    # Weighted sum
    output_bf16 = torch.bmm(attention_weights, value_bf16)
    
    # Convert back to float32
    return output_bf16.to(torch.float32)

function_signature = {
    "name": "causal_attention_bf16",
    "inputs": [
        ((8, 16, 512), torch.float32), 
        ((8, 16, 512), torch.float32), 
        ((8, 16, 512), torch.float32)
    ],
    "outputs": [
        ((8, 16, 512), torch.float32)
    ]
}
