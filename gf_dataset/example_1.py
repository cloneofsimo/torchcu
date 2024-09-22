
import torch

def torch_cosine_similarity_int8_function(input_tensor: torch.Tensor, weight: torch.Tensor, padding: int) -> torch.Tensor:
    """
    Computes the cosine similarity between two tensors after applying padding.
    """
    input_tensor = input_tensor.to(torch.int8)
    weight = weight.to(torch.int8)
    padded_input = torch.nn.functional.pad(input_tensor, (padding, padding), "constant", 0)
    similarity = torch.nn.functional.cosine_similarity(padded_input, weight, dim=1)
    return similarity.to(torch.float32)

function_signature = {
    "name": "torch_cosine_similarity_int8_function",
    "inputs": [
        ((4, 4), torch.int8),
        ((4, 4), torch.int8),
        ((), torch.int32)
    ],
    "outputs": [
        ((4,), torch.float32)
    ]
}
