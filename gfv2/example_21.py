
import torch

def torch_learned_positional_encoding(input_tensor: torch.Tensor,
                                       positional_encoding_weights: torch.Tensor) -> torch.Tensor:
    """
    Applies learned positional encoding to an input tensor.

    Args:
        input_tensor: Input tensor of shape (batch_size, seq_len, embedding_dim).
        positional_encoding_weights: Learned positional encoding weights of shape
                                      (seq_len, embedding_dim).

    Returns:
        Tensor of shape (batch_size, seq_len, embedding_dim) with learned positional
        encoding applied.
    """
    return input_tensor + positional_encoding_weights

function_signature = {
    "name": "torch_learned_positional_encoding",
    "inputs": [
        ((1, 10, 512), torch.float32),
        ((10, 512), torch.float32)
    ],
    "outputs": [
        ((1, 10, 512), torch.float32),
    ]
}
