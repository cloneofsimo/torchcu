
import torch

def pairwise_hamming_distance_add(input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Calculates the pairwise Hamming distance between each element in the input tensor 
    and the corresponding element in the weights tensor. 
    Then adds the distance to the input tensor, returning the result.

    Args:
        input_tensor: Input tensor of shape (batch_size, feature_dim).
        weights: Weights tensor of shape (feature_dim,).

    Returns:
        Tensor of shape (batch_size, feature_dim) where each element is the sum of the
        corresponding input element and its Hamming distance to the corresponding weight.
    """
    input_tensor_int8 = input_tensor.to(torch.int8)
    weights_int8 = weights.to(torch.int8)

    # Calculate pairwise Hamming distances using XOR and counting set bits
    distances = (input_tensor_int8 ^ weights_int8.unsqueeze(0)).int().sum(dim=1, keepdim=True)
    
    # Add distances to the original input tensor
    return input_tensor + distances.float()


function_signature = {
    "name": "pairwise_hamming_distance_add",
    "inputs": [
        ((10, 8), torch.float32),
        ((8,), torch.float32)
    ],
    "outputs": [
        ((10, 8), torch.float32),
    ]
}
