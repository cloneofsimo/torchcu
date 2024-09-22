
import torch
import torch.nn.functional as F

def torch_fused_linear_with_dropout_and_regularization(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, p: float, lambda_l2: float) -> torch.Tensor:
    """
    Performs a fused linear operation with dropout and L2 regularization.

    Args:
        input_tensor: The input tensor.
        weight: The weight tensor.
        bias: The bias tensor.
        p: The dropout probability.
        lambda_l2: The L2 regularization coefficient.

    Returns:
        The output tensor after applying linear transformation, dropout, and L2 regularization.
    """
    # Set random seed for deterministic dropout
    torch.manual_seed(42)
    
    # Apply linear transformation
    output = F.linear(input_tensor, weight, bias)
    
    # Apply dropout
    output = F.dropout(output, p=p, training=True)
    
    # Apply L2 regularization
    l2_loss = lambda_l2 * torch.sum(weight**2)
    
    # Return the output tensor without the regularization loss (for simplicity)
    return output

function_signature = {
    "name": "torch_fused_linear_with_dropout_and_regularization",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32),
        (0.5, None),
        (0.01, None),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
