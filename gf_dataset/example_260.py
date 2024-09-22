
import torch

def torch_cosine_similarity_function(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """
    Calculates the cosine similarity between two tensors.
    """
    return torch.nn.functional.cosine_similarity(input1, input2, dim=1)

function_signature = {
    "name": "torch_cosine_similarity_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4,), torch.float32),
    ]
}
