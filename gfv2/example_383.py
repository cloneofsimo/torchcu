
import torch
import torch.nn.functional as F

def my_complex_function(input_tensor: torch.Tensor, weights: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on the input tensor, including:
        1. Logarithm of the input tensor
        2. Matrix multiplication with weights
        3. Arcface loss calculation
        4. In-place modification of the result
    """

    # Logarithm of the input tensor
    input_tensor = torch.log(input_tensor)
    
    # Matrix multiplication
    output = torch.matmul(input_tensor, weights.t())

    # Arcface Loss Calculation
    output = F.normalize(output, p=2, dim=1)
    weights = F.normalize(weights, p=2, dim=1)
    arcface_loss = F.cosine_similarity(output, weights[labels])
    
    # In-place modification of the result
    output.mul_(arcface_loss)
    return output

function_signature = {
    "name": "my_complex_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.int64)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
