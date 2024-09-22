
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

@custom_fwd(cast_inputs=torch.bfloat16)
def torch_max_energy_computation_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value along each row and the corresponding energy.
    """
    max_values, max_indices = torch.max(input_tensor, dim=1)
    energy = torch.sum(input_tensor * torch.nn.functional.one_hot(max_indices, num_classes=input_tensor.size(1)), dim=1)
    return max_values.to(torch.float32), energy.to(torch.float32)

function_signature = {
    "name": "torch_max_energy_computation_function",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4,), torch.float32),
        ((4,), torch.float32),
    ]
}
