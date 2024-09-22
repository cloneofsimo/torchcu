
import torch

def adversarial_training_example(input_tensor: torch.Tensor, weights: torch.Tensor, perturbation: torch.Tensor) -> torch.Tensor:
    """
    Performs a simple adversarial training step using a specified perturbation.
    """
    # Add perturbation to input
    perturbed_input = input_tensor + perturbation
    # Apply weights
    output = torch.matmul(perturbed_input, weights.t())
    return output

function_signature = {
    "name": "adversarial_training_example",
    "inputs": [
        ((10, 10), torch.float32),
        ((10, 10), torch.float32),
        ((10, 10), torch.float32)
    ],
    "outputs": [
        ((10, 10), torch.float32),
    ]
}
