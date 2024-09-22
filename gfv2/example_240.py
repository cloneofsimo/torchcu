
import torch

def attention_mask_function(query: torch.Tensor, key: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked attention scores.
    """
    # Clamp query and key to prevent overflow in fp16
    query = query.clamp(-8, 8)
    key = key.clamp(-8, 8)
    # Compute attention scores
    attention_scores = torch.einsum('bhd,bhd->bht', query, key)
    # Apply mask
    attention_scores = attention_scores.masked_fill(mask == 0, -float('inf'))
    # Softmax normalization
    attention_weights = torch.softmax(attention_scores, dim=-1)
    # Return attention weights in fp32
    return attention_weights.to(torch.float32)

function_signature = {
    "name": "attention_mask_function",
    "inputs": [
        ((1, 10, 512), torch.float32),
        ((1, 10, 512), torch.float32),
        ((1, 10, 10), torch.bool)
    ],
    "outputs": [
        ((1, 10, 10), torch.float32),
    ]
}
