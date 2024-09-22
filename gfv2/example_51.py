
import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.positional_embeddings = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        # Assume x is a tensor of shape (batch_size, seq_len, d_model)
        seq_len = x.shape[1]
        return x + self.positional_embeddings[:seq_len, :]

def adaptive_avg_pool3d_hinge_loss(input_tensor: torch.Tensor, target_tensor: torch.Tensor, learned_pe: LearnedPositionalEncoding, margin: float) -> torch.Tensor:
    """
    Performs adaptive average pooling, applies learned positional encoding,
    and computes hinge embedding loss.
    """
    # Adaptive average pooling in 3D
    pooled_input = torch.nn.functional.adaptive_avg_pool3d(input_tensor, (1, 1, 1))

    # Apply learned positional encoding
    pooled_input = learned_pe(pooled_input.squeeze(dim=1))

    # Compute hinge embedding loss
    loss = torch.nn.functional.hinge_embedding_loss(pooled_input, target_tensor, margin=margin)

    return loss

function_signature = {
    "name": "adaptive_avg_pool3d_hinge_loss",
    "inputs": [
        ((2, 3, 4, 5, 6), torch.float32),  # (batch_size, channels, height, width, depth)
        ((2, 6), torch.float32),         # (batch_size, embedding_dim)
        ((5000, 6), torch.float32),      # (max_len, embedding_dim)
        (1.0, torch.float32)              # margin
    ],
    "outputs": [
        ((1,), torch.float32),           # (loss)
    ]
}
