
import torch
import torch.nn.functional as F

def flash_attn_bf16_function(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Perform flash attention with bfloat16 precision.
    """
    query = query.to(torch.bfloat16)
    key = key.to(torch.bfloat16)
    value = value.to(torch.bfloat16)

    if mask is not None:
        mask = mask.to(torch.bfloat16)

    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    attn_weights = attn_weights / torch.sqrt(torch.tensor(key.shape[-1], dtype=torch.bfloat16))

    if mask is not None:
        attn_weights.masked_fill_(~mask, -float('inf'))

    attn_weights = F.softmax(attn_weights, dim=-1)

    output = torch.matmul(attn_weights, value)
    return output.to(torch.float32)

function_signature = {
    "name": "flash_attn_bf16_function",
    "inputs": [
        ((8, 128, 64), torch.float32),
        ((8, 128, 64), torch.float32),
        ((8, 128, 64), torch.float32),
        ((8, 1, 128), torch.bool)
    ],
    "outputs": [
        ((8, 128, 64), torch.float32)
    ]
}
