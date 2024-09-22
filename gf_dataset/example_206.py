
import torch

def torch_min_gather_int8_function(input_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Finds the minimum value along a specific dimension and gathers the corresponding values from another tensor.
    """
    input_int8 = input_tensor.to(torch.int8)
    min_values, min_indices = torch.min(input_int8, dim=1)
    output = torch.gather(indices, dim=1, index=min_indices.long())
    return output.to(torch.int8)

function_signature = {
    "name": "torch_min_gather_int8_function",
    "inputs": [
        ((4, 4), torch.int8),
        ((4, 4), torch.int32)
    ],
    "outputs": [
        ((4, 1), torch.int8)
    ]
}
