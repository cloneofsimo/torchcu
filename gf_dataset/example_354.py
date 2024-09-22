
import torch
import torch.nn.functional as F
from torch.nn.functional import pad

def causal_attention_bf16(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Computes causal attention with bfloat16 precision.
    """
    # Convert to bfloat16 for faster computation
    query = query.to(torch.bfloat16)
    key = key.to(torch.bfloat16)
    value = value.to(torch.bfloat16)

    # Compute attention scores
    scores = torch.einsum('btd,bsd->bts', query, key)

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Normalize scores and apply causal mask
    scores = F.softmax(scores, dim=-1)
    scores = torch.triu(scores, diagonal=1)

    # Compute attention output
    output = torch.einsum('bts,bsd->btd', scores, value)

    # Convert back to float32 for the output
    return output.to(torch.float32)

function_signature = {
    "name": "causal_attention_bf16",
    "inputs": [
        ((1, 16, 128), torch.float32),
        ((1, 16, 128), torch.float32),
        ((1, 16, 128), torch.float32),
        ((1, 1, 128), torch.bool)
    ],
    "outputs": [
        ((1, 16, 128), torch.float32),
    ]
}
