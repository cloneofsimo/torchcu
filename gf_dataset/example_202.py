
import torch

def causal_attention_bfloat16(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    Perform causal attention with bfloat16 precision, using log_softmax for stability.
    """
    # Convert to bfloat16 for faster computation
    query_bf16 = query.to(torch.bfloat16)
    key_bf16 = key.to(torch.bfloat16)
    value_bf16 = value.to(torch.bfloat16)

    # Calculate attention scores (dot product)
    scores = torch.matmul(query_bf16, key_bf16.transpose(-2, -1)) / torch.sqrt(torch.tensor(query.shape[-1], dtype=torch.bfloat16))

    # Apply causal masking (only attend to previous tokens)
    mask = torch.triu(torch.ones(scores.shape[-2:], dtype=torch.bool), diagonal=1)
    scores.masked_fill_(mask, float('-inf'))

    # Apply log_softmax for numerical stability
    attention_weights = torch.log_softmax(scores, dim=-1)

    # Weighted sum of values
    output_bf16 = torch.matmul(attention_weights, value_bf16)

    # Convert back to float32 for output
    return output_bf16.to(torch.float32)

function_signature = {
    "name": "causal_attention_bfloat16",
    "inputs": [
        ((16, 128, 512), torch.float32),
        ((16, 128, 512), torch.float32),
        ((16, 128, 512), torch.float32)
    ],
    "outputs": [
        ((16, 128, 512), torch.float32),
    ]
}
