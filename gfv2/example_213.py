
import torch

def scatter_logspace_geglu_function(input_tensor: torch.Tensor, indices: torch.Tensor, base: float) -> torch.Tensor:
    """
    Performs a scatter operation on a logspace tensor followed by a GEGLU activation.
    """
    logspace_tensor = torch.logspace(0, 1, input_tensor.shape[1], base=base)
    scattered_tensor = torch.scatter(logspace_tensor.unsqueeze(0).repeat(input_tensor.shape[0], 1), 1, indices, input_tensor)
    geglu_output = torch.nn.functional.gelu(scattered_tensor)
    return geglu_output

function_signature = {
    "name": "scatter_logspace_geglu_function",
    "inputs": [
        ((10, 5), torch.float32),
        ((10,), torch.int64),
        (float)
    ],
    "outputs": [
        ((10, 5), torch.float32)
    ]
}
