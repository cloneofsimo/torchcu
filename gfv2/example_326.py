
import torch

def attention_pow_fp16(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, power: float) -> torch.Tensor:
    """
    Performs a simple linear attention mechanism with element-wise power operation on the attention scores.
    """
    query = query.to(torch.float16)
    key = key.to(torch.float16)
    value = value.to(torch.float16)

    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores = torch.pow(attention_scores, power)
    attention_weights = torch.softmax(attention_scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output.to(torch.float32)


function_signature = {
    "name": "attention_pow_fp16",
    "inputs": [
        ((1, 16, 16), torch.float32),
        ((1, 16, 16), torch.float32),
        ((1, 16, 16), torch.float32),
        (torch.float32),
    ],
    "outputs": [
        ((1, 16, 16), torch.float32)
    ]
}

