
import torch

def ridge_regression(input_tensor: torch.Tensor, weight: torch.Tensor, lambda_reg: float) -> torch.Tensor:
    """
    Performs ridge regression with a specified regularization strength.

    Args:
        input_tensor: Input tensor of shape (batch_size, input_features).
        weight: Weight tensor of shape (output_features, input_features).
        lambda_reg: Regularization strength.

    Returns:
        Output tensor of shape (batch_size, output_features).
    """
    output = torch.matmul(input_tensor, weight.t())
    regularization_loss = 0.5 * lambda_reg * torch.sum(weight ** 2)
    output = output - regularization_loss
    return output

function_signature = {
    "name": "ridge_regression",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        (torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
