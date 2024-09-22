
import torch

def pairwise_distance_bf16(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise Euclidean distance between two tensors using bfloat16 for efficiency.
    """
    input1_bf16 = input1.to(torch.bfloat16)
    input2_bf16 = input2.to(torch.bfloat16)

    # Calculate squared Euclidean distances
    distances_bf16 = torch.cdist(input1_bf16, input2_bf16, p=2)

    # Convert back to float for the final result
    return distances_bf16.to(torch.float32)

function_signature = {
    "name": "pairwise_distance_bf16",
    "inputs": [
        ((16, 128), torch.float32),
        ((8, 128), torch.float32)
    ],
    "outputs": [
        ((16, 8), torch.float32),
    ]
}
