
import torch
import torch.nn as nn
from torch.nn import functional as F

class MaskedAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask):
        Q = self.linear_q(query)
        K = self.linear_k(key)
        V = self.linear_v(value)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        scores = scores + mask
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to value
        output = torch.matmul(attention, V)
        return output

def masked_attention_forward_fp16(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Performs masked attention with fp16 precision and returns the result in fp32.

    Args:
        query (torch.Tensor): Query tensor with shape (batch_size, query_len, d_model).
        key (torch.Tensor): Key tensor with shape (batch_size, key_len, d_model).
        value (torch.Tensor): Value tensor with shape (batch_size, value_len, d_model).
        mask (torch.Tensor): Attention mask with shape (batch_size, query_len, key_len).

    Returns:
        torch.Tensor: Output tensor with shape (batch_size, query_len, d_model) in fp32.
    """

    query = query.to(torch.float16)
    key = key.to(torch.float16)
    value = value.to(torch.float16)
    mask = mask.to(torch.float16)

    attention = MaskedAttention(d_model=query.shape[-1])
    output = attention(query, key, value, mask)
    return output.to(torch.float32)

function_signature = {
    "name": "masked_attention_forward_fp16",
    "inputs": [
        ((1, 8, 16), torch.float32),
        ((1, 8, 16), torch.float32),
        ((1, 8, 16), torch.float32),
        ((1, 8, 8), torch.float32)
    ],
    "outputs": [
        ((1, 8, 16), torch.float32)
    ]
}
