
import torch

def logsumexp_ge_function(input_tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Calculates the log-sum-exp of the input tensor, with a threshold applied to avoid numerical issues.
    For elements exceeding the threshold, their log-sum-exp is returned directly.
    Otherwise, the log-sum-exp is calculated using the standard formula.
    """
    mask = input_tensor >= threshold
    output = torch.zeros_like(input_tensor)
    output[mask] = input_tensor[mask]
    output[~mask] = torch.logsumexp(input_tensor[~mask], dim=0)
    return output

function_signature = {
    "name": "logsumexp_ge_function",
    "inputs": [
        ((4,), torch.float32),
        (torch.float32,)
    ],
    "outputs": [
        ((4,), torch.float32),
    ]
}
