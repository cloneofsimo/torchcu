
import torch
import torch.fft
import torch.linalg as LA

def torch_low_rank_approx_fp16(input_tensor: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Performs a low-rank approximation of a matrix using SVD and then reconstructs with fp16 precision.
    """
    # Convert to fp16 for efficiency
    input_tensor_fp16 = input_tensor.to(torch.float16)

    # Compute SVD
    U, S, V = torch.linalg.svd(input_tensor_fp16)

    # Reconstruct using only the top 'rank' singular values
    S_reduced = torch.diag_embed(S[:rank])
    reconstructed_tensor = U[:, :rank] @ S_reduced @ V[:rank, :]

    # Return reconstructed tensor in fp32
    return reconstructed_tensor.to(torch.float32)

function_signature = {
    "name": "torch_low_rank_approx_fp16",
    "inputs": [
        ((10, 10), torch.float32),  # Example input shape
        (1, torch.int32)  # Rank is an integer
    ],
    "outputs": [
        ((10, 10), torch.float32),
    ]
}
