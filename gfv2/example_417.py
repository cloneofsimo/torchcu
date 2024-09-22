
import torch

def my_complex_function(input_tensor: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Performs a complex operation involving cross-entropy, scaling, and division.
    """
    input_tensor = input_tensor.to(torch.float16)
    target = target.to(torch.int8)
    weights = weights.to(torch.float16)
    
    # Calculate cross-entropy loss
    loss = torch.nn.functional.cross_entropy(input_tensor, target, weight=weights)

    # Scale loss by a factor of 2
    loss = loss * 2

    # Divide the scaled loss by the sum of weights
    loss = loss / torch.sum(weights)

    # Return the final loss value
    return loss.to(torch.float32)

function_signature = {
    "name": "my_complex_function",
    "inputs": [
        ((16, 10), torch.float32),
        ((16,), torch.int8),
        ((10,), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
