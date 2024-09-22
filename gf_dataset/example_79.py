
import torch

def torch_distance_pooling_function(input_tensor: torch.Tensor, other_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs distance-based pooling with adaptive max pooling and bfloat16 precision.
    """
    # Replicate input along the channel dimension
    input_tensor = torch.nn.functional.replication_pad1d(input_tensor, (0, 1))
    other_tensor = torch.nn.functional.replication_pad1d(other_tensor, (0, 1))

    # Convert to bfloat16
    input_bf16 = input_tensor.to(torch.bfloat16)
    other_bf16 = other_tensor.to(torch.bfloat16)

    # Calculate pairwise Manhattan distance
    distances = torch.pairwise_manhattan_distance(input_bf16, other_bf16)

    # Adaptive max pooling to reduce dimension
    pooled_distances = torch.nn.functional.adaptive_max_pool1d(distances, output_size=1)

    # Convert back to float32
    return pooled_distances.to(torch.float32)


function_signature = {
    "name": "torch_distance_pooling_function",
    "inputs": [
        ((1, 16, 4), torch.float32),  # (batch_size, num_features, seq_len)
        ((1, 16, 4), torch.float32)
    ],
    "outputs": [
        ((1, 1, 1), torch.float32)
    ]
}
