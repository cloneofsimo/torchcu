
import torch

def torch_pairwise_manhattan_distance_fp16(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """
    Calculates pairwise Manhattan distance between two tensors in fp16.
    """
    input1_fp16 = input1.to(torch.float16)
    input2_fp16 = input2.to(torch.float16)
    distances = torch.abs(input1_fp16 - input2_fp16.unsqueeze(1))
    return torch.sum(distances, dim=2, dtype=torch.float16)

function_signature = {
    "name": "torch_pairwise_manhattan_distance_fp16",
    "inputs": [
        ((16, 1024), torch.float32),
        ((16, 1024), torch.float32)
    ],
    "outputs": [
        ((16, 16), torch.float16)
    ]
}
