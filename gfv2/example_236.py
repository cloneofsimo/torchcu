
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def window_attention_fp16(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                          mask: torch.Tensor = None,
                          head_dim: int = 64,
                          qkv_bias: bool = False,
                          attn_drop: float = 0.0,
                          proj_drop: float = 0.0) -> torch.Tensor:
    """
    Window based multi-head self-attention (W-MSA) module with  fp16 precision.

    Args:
        query (torch.Tensor): Input query tensor of shape [B, N, C].
        key (torch.Tensor): Input key tensor of shape [B, N, C].
        value (torch.Tensor): Input value tensor of shape [B, N, C].
        mask (torch.Tensor, optional): Attention mask of shape [B, N, N]. Defaults to None.
        head_dim (int, optional): Dimension of each attention head. Defaults to 64.
        qkv_bias (bool, optional): Whether to add bias to qkv. Defaults to False.
        attn_drop (float, optional): Dropout rate for attention weights. Defaults to 0.0.
        proj_drop (float, optional): Dropout rate for output projection. Defaults to 0.0.

    Returns:
        torch.Tensor: Output tensor of shape [B, N, C].
    """
    B, N, C = query.shape
    assert C % head_dim == 0
    num_heads = C // head_dim

    qkv = torch.cat([query, key, value], dim=2).to(torch.float16)  # [B, N, 3C]
    qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
    q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

    q = q * (head_dim ** -0.5)
    attn = (q @ k.transpose(-2, -1))  # [B, num_heads, N, N]
    
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float('-inf'))
    
    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attn_drop, training=self.training)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C).to(torch.float32)
    x = F.dropout(x, p=proj_drop, training=self.training)

    return x

function_signature = {
    "name": "window_attention_fp16",
    "inputs": [
        ((1, 8, 256), torch.float32),
        ((1, 8, 256), torch.float32),
        ((1, 8, 256), torch.float32),
        ((1, 8, 8), torch.bool), 
    ],
    "outputs": [
        ((1, 8, 256), torch.float32),
    ]
}
