
import torch

def kronecker_celu_function(input_tensor: torch.Tensor, weight: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Computes the Kronecker product of the input tensor with a weight tensor, 
    applies the CELU activation function, and then returns the result.
    """
    kronecker_product = torch.kron(input_tensor, weight)
    celu_output = torch.nn.functional.celu(kronecker_product, alpha=alpha)
    return celu_output

function_signature = {
    "name": "kronecker_celu_function",
    "inputs": [
        ((1,), torch.float32),
        ((2, 2), torch.float32),
        (1.0, torch.float32)
    ],
    "outputs": [
        ((2,), torch.float32),
    ]
}
