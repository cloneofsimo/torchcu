
import torch

def multinomial_min_function(input_tensor: torch.Tensor, num_samples: int, probabilities: torch.Tensor) -> torch.Tensor:
    """
    This function performs the following operations:
    1. Calculates the multinomial distribution of the input tensor based on the probabilities.
    2. Finds the minimum value of the generated samples.
    3. Backpropagates the gradients through the function.
    """
    samples = torch.multinomial(probabilities, num_samples=num_samples, replacement=True)
    min_values = torch.gather(input_tensor, dim=1, index=samples.unsqueeze(-1)).squeeze()
    min_values.backward(torch.ones_like(min_values))
    return min_values

function_signature = {
    "name": "multinomial_min_function",
    "inputs": [
        ((10,), torch.float32),
        (1, torch.int32),
        ((10,), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
