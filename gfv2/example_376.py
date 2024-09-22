
import torch

def my_function(input_tensor: torch.Tensor, weights: torch.Tensor, regularization_param: float) -> torch.Tensor:
    """
    Performs a simple linear transformation, applies a custom activation, and performs lasso regularization.
    """
    output = torch.einsum('ij,jk->ik', input_tensor, weights)
    output = torch.sigmoid(output) 
    # Custom activation function
    output = output * (1 + torch.abs(output)) # Example custom activation
    regularization = regularization_param * torch.sum(torch.abs(weights))
    output = output - regularization 
    return output

function_signature = {
    "name": "my_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
