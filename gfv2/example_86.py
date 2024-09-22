
import torch

def custom_loss_function(input_tensor: torch.Tensor, target_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Calculates a custom loss function combining binary cross-entropy, a linear transformation, and margin ranking loss.

    Args:
        input_tensor: The input tensor of shape (batch_size, features).
        target_tensor: The target tensor of shape (batch_size).
        weights: The weights tensor of shape (features).

    Returns:
        A tensor representing the computed loss value.
    """

    # Convert to bfloat16 for potential speedup
    input_tensor = input_tensor.to(torch.bfloat16)
    weights = weights.to(torch.bfloat16)

    # Calculate binary cross-entropy
    bce_loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(input_tensor), target_tensor.float())

    # Apply linear transformation with weights
    linear_output = torch.addmv(input_tensor, torch.zeros_like(input_tensor), weights)

    # Calculate margin ranking loss for pairwise comparison
    margin_loss = torch.nn.functional.margin_ranking_loss(
        linear_output,
        torch.ones_like(linear_output),
        margin=1.0,
        reduction='mean'
    )

    # Combine losses and return in fp32
    combined_loss = bce_loss + margin_loss
    return combined_loss.to(torch.float32)

function_signature = {
    "name": "custom_loss_function",
    "inputs": [
        ((4, 8), torch.float32),  # (batch_size, features)
        ((4,), torch.float32),  # (batch_size)
        ((8,), torch.float32)  # (features)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
