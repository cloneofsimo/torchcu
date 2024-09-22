
import torch
import torch.nn.functional as F
from cutlass import *

def torch_outer_product_standardization(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Calculates the outer product of input and weight, 
    standardizes the weight along the last dimension, and applies ReLU activation.
    """
    # Calculate the outer product
    outer_product = torch.einsum('ij,ik->ijk', input_tensor, weight)

    # Standardize the weight along the last dimension
    weight_mean = weight.mean(dim=-1, keepdim=True)
    weight_std = weight.std(dim=-1, keepdim=True)
    standardized_weight = (weight - weight_mean) / weight_std

    # Apply ReLU activation
    output = F.relu(outer_product * standardized_weight)

    return output

function_signature = {
    "name": "torch_outer_product_standardization",
    "inputs": [
        ((2, 3), torch.float32),
        ((3, 4), torch.float32)
    ],
    "outputs": [
        ((2, 3, 4), torch.float32)
    ]
}
