
import torch

def index_select_function(input_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Performs index selection on the input tensor using the given indices.
    """
    output = torch.index_select(input_tensor, dim=1, index=indices)
    return output

function_signature = {
    "name": "index_select_function",
    "inputs": [
        ((10, 5), torch.float32),
        ((10,), torch.int64)
    ],
    "outputs": [
        ((10, 1), torch.float32),
    ]
}
