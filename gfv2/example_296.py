
import torch

def relative_positional_encoding_int8(input_tensor: torch.Tensor,
                                    seq_len: int,
                                    max_relative_positions: int,
                                    device: torch.device = torch.device("cuda")):
    """
    Computes relative positional encoding for a tensor of integer values.

    Args:
        input_tensor: A tensor of shape (batch_size, seq_len) representing integer values.
        seq_len: The length of the sequence.
        max_relative_positions: The maximum number of relative positions to consider.
        device: The device to use for computations.

    Returns:
        A tensor of shape (batch_size, seq_len, seq_len) representing the relative positional encodings.
    """
    assert input_tensor.dtype == torch.int8
    batch_size = input_tensor.size(0)

    # Compute pairwise Hamming distances
    distances = pairwise_hamming_distance(input_tensor, input_tensor)

    # Create relative positional encodings
    relative_positions = torch.arange(-max_relative_positions, max_relative_positions + 1, device=device)
    relative_position_embeddings = torch.zeros((2 * max_relative_positions + 1, seq_len, seq_len), device=device)
    for i in range(seq_len):
        for j in range(seq_len):
            relative_position_embeddings[distances[i, j] + max_relative_positions, i, j] = 1.0

    # Broadcast relative positional embeddings to batch size
    relative_position_embeddings = relative_position_embeddings.expand(batch_size, -1, -1)

    return relative_position_embeddings.to(torch.float16)

def pairwise_hamming_distance(x, y):
    """
    Computes the pairwise Hamming distance between two tensors.

    Args:
        x: A tensor of shape (batch_size, seq_len).
        y: A tensor of shape (batch_size, seq_len).

    Returns:
        A tensor of shape (seq_len, seq_len) representing the pairwise Hamming distances.
    """
    assert x.dtype == torch.int8
    assert y.dtype == torch.int8
    return (x[:, None, :] != y[None, :, :]).sum(-1)

function_signature = {
    "name": "relative_positional_encoding_int8",
    "inputs": [
        ((1, 10), torch.int8),
        (10, ),
        (10, ),
    ],
    "outputs": [
        ((1, 10, 10), torch.float16),
    ]
}
