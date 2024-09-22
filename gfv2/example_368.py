
import torch

def hamming_distance_layer(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Calculates pairwise Hamming distance between input tensor and weight tensor.

    Args:
        input_tensor: Tensor of shape (N, D), where N is the batch size and D is the feature dimension.
        weight: Tensor of shape (M, D), where M is the number of weight vectors.

    Returns:
        Tensor of shape (N, M), where each element represents the Hamming distance between the corresponding
        input vector and weight vector.
    """

    # Convert tensors to binary representation (0 or 1)
    input_tensor = (input_tensor > 0).float()
    weight = (weight > 0).float()

    # Calculate pairwise Hamming distances using broadcasting
    distances = torch.sum(torch.abs(input_tensor[:, None, :] - weight[None, :, :]), dim=2)

    return distances

function_signature = {
    "name": "hamming_distance_layer",
    "inputs": [
        ((4, 8), torch.float32),
        ((16, 8), torch.float32),
    ],
    "outputs": [
        ((4, 16), torch.float32),
    ]
}
