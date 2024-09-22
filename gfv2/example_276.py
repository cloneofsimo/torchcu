
import torch
from torch.nn.functional import softmax

def causal_attention_sparse_bf16_function(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, sparsity_ratio: float) -> torch.Tensor:
    """
    Performs causal attention with bfloat16 precision and sparse training.

    Args:
        query: Query tensor (B, T, H) where B is batch size, T is sequence length, H is hidden dimension.
        key: Key tensor (B, T, H).
        value: Value tensor (B, T, H).
        mask: Causal mask tensor (B, T, T).
        sparsity_ratio: Sparsity ratio for training (float between 0 and 1).

    Returns:
        Output tensor (B, T, H) after causal attention.
    """

    # Convert to bfloat16 for computation
    query_bf16 = query.to(torch.bfloat16)
    key_bf16 = key.to(torch.bfloat16)
    value_bf16 = value.to(torch.bfloat16)

    # Sparse training: randomly drop a portion of the attention weights
    if sparsity_ratio > 0:
        attention_weights = torch.matmul(query_bf16, key_bf16.transpose(1, 2)) / (query_bf16.shape[-1] ** 0.5)
        attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))
        attention_weights = softmax(attention_weights, dim=-1)
        random_mask = torch.rand(attention_weights.shape) < sparsity_ratio
        attention_weights.masked_fill_(random_mask, 0.0)
    else:
        attention_weights = torch.matmul(query_bf16, key_bf16.transpose(1, 2)) / (query_bf16.shape[-1] ** 0.5)
        attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))
        attention_weights = softmax(attention_weights, dim=-1)

    # Attention calculation
    output_bf16 = torch.matmul(attention_weights, value_bf16)
    output = output_bf16.to(torch.float32)

    return output

function_signature = {
    "name": "causal_attention_sparse_bf16_function",
    "inputs": [
        ((2, 10, 512), torch.float32),
        ((2, 10, 512), torch.float32),
        ((2, 10, 512), torch.float32),
        ((2, 10, 10), torch.bool),
        (0.2, torch.float32)
    ],
    "outputs": [
        ((2, 10, 512), torch.float32),
    ]
}
