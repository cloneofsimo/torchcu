
import torch

def sparse_linear_bfloat16_function(input_tensor: torch.Tensor, weight: torch.Tensor, pruning_mask: torch.Tensor) -> torch.Tensor:
    """
    Applies a sparse linear transformation (matrix multiplication) with pruning mask, using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    pruning_mask_bf16 = pruning_mask.to(torch.bfloat16)

    # Apply pruning mask
    masked_weight_bf16 = weight_bf16 * pruning_mask_bf16

    # Sparse matrix multiplication
    output_bf16 = torch.matmul(input_bf16, masked_weight_bf16.t())

    # ReLU activation
    output_bf16 = torch.relu(output_bf16, inplace=True)

    # Convert back to float32
    output = output_bf16.to(torch.float32)
    return output

function_signature = {
    "name": "sparse_linear_bfloat16_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
