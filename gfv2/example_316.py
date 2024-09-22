
import torch
import torch.nn.functional as F

def window_attention_with_bfloat16_and_gradient_clipping(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    window_size: int,
    head_dim: int,
    relative_position_bias: torch.Tensor,
    gradient_clip_value: float = 1.0,
) -> torch.Tensor:
    """
    Computes window-based self-attention with bfloat16 precision and gradient clipping.

    Args:
        q: Query tensor of shape (B, N, head_dim), where B is batch size, N is sequence length.
        k: Key tensor of shape (B, N, head_dim).
        v: Value tensor of shape (B, N, head_dim).
        attn_mask: Attention mask of shape (B, N, N), indicating valid positions.
        window_size: Size of the window for attention.
        head_dim: Dimension of each attention head.
        relative_position_bias: Relative position bias tensor of shape (2 * window_size - 1, 2 * window_size - 1).
        gradient_clip_value: Value for gradient clipping.

    Returns:
        Attention output tensor of shape (B, N, head_dim).
    """

    # Convert to bfloat16 for faster computation
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)

    # Reshape for windowed attention
    B, N, _ = q.shape
    q = q.view(B, N // window_size, window_size, head_dim)
    k = k.view(B, N // window_size, window_size, head_dim)
    v = v.view(B, N // window_size, window_size, head_dim)
    attn_mask = attn_mask.view(B, N // window_size, window_size, window_size)

    # Calculate attention scores
    attn_scores = torch.einsum('bhwk,bhwm->bhkm', q, k) / head_dim**0.5
    attn_scores = attn_scores + relative_position_bias.view(1, 1, window_size, window_size)
    attn_scores = torch.where(attn_mask == 0, -1e9, attn_scores)

    # Softmax normalization
    attn_weights = F.softmax(attn_scores, dim=-1)

    # Calculate output
    output = torch.einsum('bhkm,bhwm->bhwk', attn_weights, v)
    output = output.view(B, N, head_dim)

    # Convert back to float32 and apply gradient clipping
    output = output.to(torch.float32)
    output.register_hook(lambda grad: torch.clamp(grad, -gradient_clip_value, gradient_clip_value))

    return output

function_signature = {
    "name": "window_attention_with_bfloat16_and_gradient_clipping",
    "inputs": [
        ((1, 16, 64), torch.float32),
        ((1, 16, 64), torch.float32),
        ((1, 16, 64), torch.float32),
        ((1, 4, 4, 4), torch.bool),
        (4, ),
        (64, ),
        ((7, 7), torch.float32),
        (1.0, ),
    ],
    "outputs": [
        ((1, 16, 64), torch.float32),
    ]
}
