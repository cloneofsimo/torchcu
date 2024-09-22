
import torch
import torch.nn.functional as F

def torch_lasso_relu_function(input_tensor: torch.Tensor, weight: torch.Tensor, lambda_value: float) -> torch.Tensor:
    """
    Applies lasso regularization (L1 penalty) and ReLU activation to a linear transformation.
    """
    output = F.linear(input_tensor, weight)
    output = F.relu(output)
    output = torch.where(torch.abs(output) > lambda_value, output - lambda_value, torch.zeros_like(output))
    return output

function_signature = {
    "name": "torch_lasso_relu_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        (torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
