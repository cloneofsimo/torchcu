
import torch

def window_attention_bf16(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Window-based attention using bfloat16.
    """
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)

    attn = torch.bmm(q, k.transpose(1, 2))
    attn = attn / (q.shape[-1] ** 0.5)
    if mask is not None:
        attn = attn.masked_fill(mask, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    out = torch.bmm(attn, v)

    return out.to(torch.float32)

function_signature = {
    "name": "window_attention_bf16",
    "inputs": [
        ((2, 32, 128), torch.float32),
        ((2, 32, 128), torch.float32),
        ((2, 32, 128), torch.float32),
        ((2, 32, 32), torch.bool)
    ],
    "outputs": [
        ((2, 32, 128), torch.float32),
    ]
}
