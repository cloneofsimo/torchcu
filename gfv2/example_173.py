
import torch
import torch.nn as nn
from typing import List

def example_function(input_tensor: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> List[torch.Tensor]:
    """
    Performs a series of operations including diagonal extraction, binary cross-entropy, 
    adaptive log softmax, negative log likelihood loss, and backpropagation.
    """

    # Convert to fp16 for potential speedup (if supported)
    input_tensor = input_tensor.to(torch.float16)
    target = target.to(torch.float16)
    weights = weights.to(torch.float16)

    # Extract diagonal
    diagonal = torch.diag(input_tensor)

    # Binary cross-entropy with weights
    bce_loss = nn.functional.binary_cross_entropy_with_logits(input_tensor, target, weights)

    # Adaptive log softmax
    log_softmax = nn.functional.adaptive_log_softmax(input_tensor, dim=1)

    # Negative log likelihood loss
    nll_loss = nn.functional.nll_loss(log_softmax, target, reduction='mean')

    # Backpropagation
    nll_loss.backward()

    # Return results (converted back to fp32)
    return [
        diagonal.to(torch.float32),
        bce_loss.to(torch.float32),
        log_softmax.to(torch.float32),
        nll_loss.to(torch.float32),
        input_tensor.grad.to(torch.float32)  # Gradients for input
    ]

function_signature = {
    "name": "example_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4,), torch.int64),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((4,), torch.float32),
        ((), torch.float32),
        ((4, 4), torch.float32),
        ((), torch.float32),
        ((4, 4), torch.float32),
    ]
}
