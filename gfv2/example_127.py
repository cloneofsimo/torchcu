
import torch
import torch.nn.functional as F

def masked_attention_fp16(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Masked attention layer with FP16 computation.

    Args:
        query: Query tensor (batch_size, seq_len, embedding_dim)
        key: Key tensor (batch_size, seq_len, embedding_dim)
        value: Value tensor (batch_size, seq_len, embedding_dim)
        mask: Attention mask (batch_size, seq_len)

    Returns:
        Output tensor (batch_size, seq_len, embedding_dim)
    """
    query = query.to(torch.float16)
    key = key.to(torch.float16)
    value = value.to(torch.float16)
    
    attention_scores = torch.bmm(query, key.transpose(1, 2)) / (query.size(-1) ** 0.5)
    attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(attention_scores, dim=-1)
    
    output = torch.bmm(attention_weights, value)
    output = output.to(torch.float32)
    return output

function_signature = {
    "name": "masked_attention_fp16",
    "inputs": [
        ((1, 4, 8), torch.float32),
        ((1, 4, 8), torch.float32),
        ((1, 4, 8), torch.float32),
        ((1, 4), torch.bool)
    ],
    "outputs": [
        ((1, 4, 8), torch.float32)
    ]
}

