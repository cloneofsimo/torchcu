
import torch

def adaptive_softmax_function(input_tensor: torch.Tensor, weight: torch.Tensor, layer_scaling: float) -> torch.Tensor:
    """
    Performs adaptive softmax with layer scaling.
    """
    input_tensor = input_tensor * layer_scaling
    output = torch.nn.functional.adaptive_log_softmax(input_tensor, dim=-1)
    return output

function_signature = {
    "name": "adaptive_softmax_function",
    "inputs": [
        ((10, 5), torch.float32),
        ((5, 10), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((10, 10), torch.float32)
    ]
}
