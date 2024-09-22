
import torch
import torch.nn.functional as F

def attention_weighted_average(
    input_tensor: torch.Tensor,
    attention_weights: torch.Tensor,
    fading_in_factor: float = 1.0,
) -> torch.Tensor:
    """
    Calculates a weighted average of the input tensor using attention weights,
    with a fading-in factor applied.
    """
    # Apply weight standardization
    attention_weights_standardized = F.softmax(attention_weights, dim=-1)

    # Apply fading-in
    attention_weights_standardized *= fading_in_factor

    # Perform weighted average
    weighted_average = torch.sum(input_tensor * attention_weights_standardized.unsqueeze(-1), dim=1)

    return weighted_average.to(torch.float16)

function_signature = {
    "name": "attention_weighted_average",
    "inputs": [
        ((1, 10, 16), torch.float32),
        ((1, 10), torch.float32),
        (1.0, torch.float32)
    ],
    "outputs": [
        ((1, 16), torch.float16),
    ]
}

