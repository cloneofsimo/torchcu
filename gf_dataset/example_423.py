
import torch

def pairwise_hamming_distance_fp16(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """
    Calculates the pairwise Hamming distance between two tensors of boolean values.
    
    Args:
        input1: A tensor of boolean values with shape (batch_size, num_features).
        input2: A tensor of boolean values with shape (batch_size, num_features).

    Returns:
        A tensor of pairwise Hamming distances with shape (batch_size, batch_size).
    """
    input1_fp16 = input1.to(torch.float16)
    input2_fp16 = input2.to(torch.float16)
    distance = torch.sum(torch.abs(input1_fp16 - input2_fp16), dim=1)
    return distance.to(torch.float32)

function_signature = {
    "name": "pairwise_hamming_distance_fp16",
    "inputs": [
        ((16, 128), torch.bool),
        ((16, 128), torch.bool)
    ],
    "outputs": [
        ((16, 16), torch.float32),
    ]
}
