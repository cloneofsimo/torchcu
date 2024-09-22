
import torch

def complex_function(input_tensor: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a complex computation involving baddbmm, masked_select, and a final reduction.
    """
    # Matrix multiplication
    output = torch.baddbmm(torch.zeros_like(input_tensor), input_tensor, weight.t(), beta=1.0, alpha=1.0)

    # Masked selection
    masked_output = torch.masked_select(output, mask)

    # Reduction
    result = torch.sum(masked_output)

    return result

function_signature = {
    "name": "complex_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.bool),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}

