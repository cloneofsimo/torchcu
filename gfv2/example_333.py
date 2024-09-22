
import torch

def ridge_regression_int8(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                         lambda_value: float) -> torch.Tensor:
    """
    Performs ridge regression with int8 precision and returns the prediction as a single int8 value.
    """
    input_tensor_int8 = input_tensor.to(torch.int8)
    weight_int8 = weight.to(torch.int8)
    bias_int8 = bias.to(torch.int8)
    
    prediction = torch.matmul(input_tensor_int8, weight_int8.t()) + bias_int8
    
    # Ridge regularization
    regularization_term = lambda_value * torch.sum(weight_int8.pow(2))
    
    prediction -= regularization_term.to(torch.int8)
    
    # Return the first element as int8
    return prediction[0].to(torch.int8)

function_signature = {
    "name": "ridge_regression_int8",
    "inputs": [
        ((1, 4), torch.float32),
        ((4, 1), torch.float32),
        ((1,), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((1,), torch.int8)
    ]
}
