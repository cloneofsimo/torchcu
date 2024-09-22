
import torch

def torch_group_norm_function(input_tensor: torch.Tensor, num_groups: int) -> torch.Tensor:
    """
    Performs group normalization on the input tensor.
    """
    return torch.nn.functional.group_norm(input_tensor, num_groups=num_groups)

function_signature = {
    "name": "torch_group_norm_function",
    "inputs": [
        ((4, 4, 4, 4), torch.float32),
        (1, torch.int32)
    ],
    "outputs": [
        ((4, 4, 4, 4), torch.float32),
    ]
}
