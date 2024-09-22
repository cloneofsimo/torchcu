
import torch
from cutlass import *

def torch_svd_hamming_similarity(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the singular value decomposition (SVD) of the input tensor and then
    computes the pairwise Hamming distance between the singular values and the target tensor.
    The function uses bfloat16 for computation to reduce memory consumption.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    target_bf16 = target_tensor.to(torch.bfloat16)

    u, s, vh = torch.linalg.svd(input_bf16)
    s_bf16 = s.to(torch.bfloat16)

    hamming_distance = torch.zeros((s_bf16.shape[0], target_bf16.shape[0]), dtype=torch.float32)
    for i, s_val in enumerate(s_bf16):
        for j, t_val in enumerate(target_bf16):
            hamming_distance[i, j] = torch.sum(torch.bitwise_xor(s_val.to(torch.int8), t_val.to(torch.int8)))

    similarity = 1.0 - hamming_distance / (s_val.shape[0] * 8)
    return similarity.to(torch.float32)

function_signature = {
    "name": "torch_svd_hamming_similarity",
    "inputs": [
        ((4, 4), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
