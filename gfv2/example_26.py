
import torch

def masked_attention_bf16(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Performs masked attention with bfloat16 precision for efficiency.
    """
    query = query.to(torch.bfloat16)
    key = key.to(torch.bfloat16)
    value = value.to(torch.bfloat16)
    mask = mask.to(torch.bfloat16)

    # Calculate attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / (query.shape[-1] ** 0.5)

    # Apply mask
    scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax over the last dimension
    attention_weights = torch.softmax(scores, dim=-1)

    # Multiply attention weights with values
    output = torch.matmul(attention_weights, value)

    # Return output in float32
    return output.to(torch.float32)


function_signature = {
    "name": "masked_attention_bf16",
    "inputs": [
        ((16, 128, 64), torch.float32),  # query
        ((16, 128, 64), torch.float32),  # key
        ((16, 128, 64), torch.float32),  # value
        ((16, 128, 128), torch.float32)   # mask
    ],
    "outputs": [
        ((16, 128, 64), torch.float32)
    ]
}
