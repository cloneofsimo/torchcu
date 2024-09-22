
import torch
import torch.nn.functional as F

def pairwise_manhattan_distance_cuda(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise Manhattan distance between two tensors.

    Args:
        input1: (N, D) tensor.
        input2: (M, D) tensor.

    Returns:
        (N, M) tensor containing the pairwise Manhattan distances.
    """
    input1_expanded = input1.unsqueeze(1).expand(-1, input2.shape[0], -1)  # (N, M, D)
    input2_expanded = input2.unsqueeze(0).expand(input1.shape[0], -1, -1)  # (N, M, D)
    distances = torch.abs(input1_expanded - input2_expanded).sum(dim=-1)  # (N, M)
    return distances

function_signature = {
    "name": "pairwise_manhattan_distance_cuda",
    "inputs": [
        ((10, 5), torch.float32),
        ((20, 5), torch.float32)
    ],
    "outputs": [
        ((10, 20), torch.float32)
    ]
}
