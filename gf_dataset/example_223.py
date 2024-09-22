
import torch
import torch.nn.functional as F

def pairwise_hamming_distance_clipped(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Calculates the pairwise Hamming distance between input and weight, applies gradient clipping, and returns in fp32.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    
    # Calculate Hamming distance
    distances = torch.sum(torch.abs(input_bf16.unsqueeze(1) - weight_bf16.unsqueeze(0)), dim=-1)

    # Gradient clipping
    distances = torch.clamp(distances, min=0.0, max=10.0)  # Example clipping range

    return distances.to(torch.float32)

function_signature = {
    "name": "pairwise_hamming_distance_clipped",
    "inputs": [
        ((8, 16), torch.float32),
        ((4, 16), torch.float32)
    ],
    "outputs": [
        ((8, 4), torch.float32),
    ]
}
