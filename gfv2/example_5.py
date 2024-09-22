
import torch
import torch.nn.functional as F

def sparse_local_attention_fp16(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor,
                                  window_size: int, sparsity: float) -> torch.Tensor:
    """
    Performs sparse local attention with FP16 precision.

    Args:
        query: Query tensor of shape (B, N, H).
        key: Key tensor of shape (B, N, H).
        value: Value tensor of shape (B, N, H).
        mask: Attention mask tensor of shape (B, N, N).
        window_size: Size of the local attention window.
        sparsity: Sparsity level for the attention weights (between 0 and 1).

    Returns:
        Output tensor of shape (B, N, H).
    """

    # Convert tensors to FP16
    query = query.to(torch.float16)
    key = key.to(torch.float16)
    value = value.to(torch.float16)

    # Compute attention scores
    scores = torch.matmul(query, key.transpose(1, 2)) / (key.shape[-1] ** 0.5)

    # Apply local attention window
    scores = scores.masked_fill(mask == 0, float('-inf'))
    scores = F.pad(scores, (window_size // 2, window_size // 2))
    scores = scores[:, :, window_size // 2: -window_size // 2]

    # Apply sparsity
    scores = scores.sort(dim=-1, descending=True)
    num_sparse_weights = int(scores.shape[-1] * sparsity)
    scores = scores[:, :, :num_sparse_weights]
    scores = scores.scatter(dim=-1, index=scores.argsort(dim=-1), src=scores)

    # Normalize attention weights
    attention_weights = torch.softmax(scores, dim=-1)

    # Compute weighted sum
    output = torch.matmul(attention_weights, value)

    return output.to(torch.float32)

function_signature = {
    "name": "sparse_local_attention_fp16",
    "inputs": [
        ((8, 128, 512), torch.float32),
        ((8, 128, 512), torch.float32),
        ((8, 128, 512), torch.float32),
        ((8, 128, 128), torch.bool),
        (16, ), torch.int32,
        (0.5, ), torch.float32
    ],
    "outputs": [
        ((8, 128, 512), torch.float32),
    ]
}
