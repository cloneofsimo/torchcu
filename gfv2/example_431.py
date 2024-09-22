
import torch

def pruned_linear_int8(input_tensor: torch.Tensor, weight: torch.Tensor, pruning_mask: torch.Tensor) -> torch.Tensor:
    """
    Performs a linear transformation with pruning applied, then quantizes the output to int8.
    """
    # Apply pruning mask to the weight
    pruned_weight = weight * pruning_mask

    # Perform linear transformation
    output = torch.matmul(input_tensor, pruned_weight.t())

    # Quantize to int8
    output_int8 = torch.quantize_per_tensor(output, scale=1.0, zero_point=0, dtype=torch.qint8)

    return output_int8

function_signature = {
    "name": "pruned_linear_int8",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.bool)
    ],
    "outputs": [
        ((4, 4), torch.qint8),
    ]
}
