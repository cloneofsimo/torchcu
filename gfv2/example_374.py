
import torch

def my_complex_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs adaptive average pooling, applies sigmoid activation, 
    and multiplies by a scalar value.
    """
    output = torch.nn.functional.adaptive_avg_pool2d(input_tensor, (1, 1))  # Adaptive average pooling
    output = torch.sigmoid(output)  # Sigmoid activation
    output = output * 2.5  # Scalar multiplication
    return output

function_signature = {
    "name": "my_complex_function",
    "inputs": [
        ((1, 3, 10, 10), torch.float32)
    ],
    "outputs": [
        ((1, 1, 1, 1), torch.float32)
    ]
}
