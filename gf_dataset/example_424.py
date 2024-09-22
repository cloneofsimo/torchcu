
import torch
from torch import nn
from torch.nn.functional import avg_pool1d

def causal_attention_int8_example(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    Performs causal self-attention with int8 quantization and a custom attention mask.
    """
    # Int8 Quantization
    query = query.to(torch.int8)
    key = key.to(torch.int8)
    value = value.to(torch.int8)

    # Causal Mask
    batch_size, seq_len = query.shape[:2]
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

    # Attention Scores
    attention = torch.einsum("bhid,bhjd->bhij", query, key)

    # Apply Mask and Softmax
    attention.masked_fill_(mask, float('-inf'))
    attention = torch.softmax(attention, dim=-1)

    # Weighted Sum
    output = torch.einsum("bhij,bhjd->bhid", attention, value)

    # Dequantize
    output = output.to(torch.float32)

    # Average Pooling
    output = avg_pool1d(output, kernel_size=2, stride=2)  # In-place operation

    return output

function_signature = {
    "name": "causal_attention_int8_example",
    "inputs": [
        ((8, 16, 32), torch.float32),  # Query
        ((8, 16, 32), torch.float32),  # Key
        ((8, 16, 32), torch.float32)   # Value
    ],
    "outputs": [
        ((8, 8, 32), torch.float32),  # Output after pooling
    ]
}
