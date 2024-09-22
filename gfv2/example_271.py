
import torch

def low_rank_approx_pow(input_tensor: torch.Tensor, rank: int, exponent: float) -> torch.Tensor:
    """
    Performs a low-rank approximation of the input tensor, then applies an element-wise power operation.

    Args:
        input_tensor: The input tensor to approximate.
        rank: The rank of the approximation.
        exponent: The exponent to apply to the approximated tensor.

    Returns:
        The approximated and powered tensor.
    """
    # Perform low-rank approximation using SVD
    U, S, V = torch.linalg.svd(input_tensor)
    S_reduced = torch.diag(S[:rank])
    approx_tensor = U[:, :rank] @ S_reduced @ V[:rank, :]

    # Apply element-wise power
    powered_tensor = torch.pow(approx_tensor, exponent)

    return powered_tensor

function_signature = {
    "name": "low_rank_approx_pow",
    "inputs": [
        ((10, 10), torch.float32),  # Input tensor
        ((), torch.int32),          # Rank
        ((), torch.float32)         # Exponent
    ],
    "outputs": [
        ((10, 10), torch.float32),  # Output tensor
    ]
}
