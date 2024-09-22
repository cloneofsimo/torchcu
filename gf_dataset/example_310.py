
import torch

def torch_elu_function(input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Applies the ELU function element-wise.
    """
    return torch.nn.functional.elu(input_tensor, alpha=alpha)

function_signature = {
    "name": "torch_elu_function",
    "inputs": [
        ((4, 4), torch.float32),
        (torch.float32,)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
