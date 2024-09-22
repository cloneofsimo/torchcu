
import torch

def self_attention_inplace(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    Performs self-attention inplace on the provided query, key, and value tensors.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / d_k ** 0.5
    attention = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention, value)
    query.copy_(output)
    return query

function_signature = {
    "name": "self_attention_inplace",
    "inputs": [
        ((16, 128, 64), torch.float32),
        ((16, 128, 64), torch.float32),
        ((16, 128, 64), torch.float32)
    ],
    "outputs": [
        ((16, 128, 64), torch.float32),
    ]
}
