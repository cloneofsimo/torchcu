
import torch

def hamming_distance_int8(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the pairwise Hamming distance between two tensors of type int8.
    """
    input1_int8 = input1.to(torch.int8)
    input2_int8 = input2.to(torch.int8)
    
    # XOR the tensors to get the bits that differ
    diff = input1_int8 ^ input2_int8 
    
    # Count the set bits using popcount
    distances = torch.bitwise_not(diff).type(torch.int8).sum(axis=1)
    
    return distances.to(torch.int32)

function_signature = {
    "name": "hamming_distance_int8",
    "inputs": [
        ((4, 4), torch.int8),
        ((4, 4), torch.int8),
    ],
    "outputs": [
        ((4,), torch.int32),
    ]
}
