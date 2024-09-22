
import torch
import torch.nn.functional as F
from torch.distributions import Gumbel

def gumbel_softmax_function(input_tensor: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    """
    Applies Gumbel-Softmax to the input tensor, allowing sampling from a categorical distribution with gradients.
    """
    gumbel_noise = Gumbel(torch.zeros_like(input_tensor), torch.ones_like(input_tensor)).sample()
    log_probs = F.log_softmax(input_tensor, dim=-1) + gumbel_noise
    probs = torch.exp(log_probs / temperature)

    if hard:
        one_hot_probs = torch.zeros_like(probs).scatter(-1, torch.argmax(probs, dim=-1, keepdim=True), 1)
        return one_hot_probs
    else:
        return probs

function_signature = {
    "name": "gumbel_softmax_function",
    "inputs": [
        ((1,), torch.float32),
        (1, torch.float32),
        (False, torch.bool)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
