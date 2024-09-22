
import torch

def pairwise_manhattan_distance_fp16(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """
    Calculate pairwise Manhattan distances between two tensors using FP16 precision.
    """
    input1_fp16 = input1.to(torch.float16)
    input2_fp16 = input2.to(torch.float16)
    # Use broadcasting to calculate pairwise distances
    distances = torch.abs(input1_fp16.unsqueeze(1) - input2_fp16.unsqueeze(0)).sum(dim=-1)
    return distances.to(torch.float32)

function_signature = {
    "name": "pairwise_manhattan_distance_fp16",
    "inputs": [
        ((10, 5), torch.float32),
        ((12, 5), torch.float32)
    ],
    "outputs": [
        ((10, 12), torch.float32),
    ]
}
