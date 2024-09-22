
import torch
from torch.nn import LayerNorm

def torch_scatter_add_softsign_layer_norm_int8(input_tensor: torch.Tensor, indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a scatter add operation, followed by a softsign activation, and then layer normalization, using int8 precision for efficiency.
    """
    # Convert to int8
    input_int8 = input_tensor.to(torch.int8)
    indices_int32 = indices.to(torch.int32)  # Indices need to be int32 for scatter_add
    weight_int8 = weight.to(torch.int8)

    # Scatter add
    output_int8 = torch.zeros_like(input_int8)
    output_int8.scatter_add_(0, indices_int32, input_int8)

    # Softsign activation
    output_int8 = (output_int8.to(torch.float32) / (1 + output_int8.to(torch.float32).abs())).to(torch.int8)

    # Layer normalization
    output_int8 = LayerNorm(output_int8.shape[1:], elementwise_affine=False)(output_int8.to(torch.float32)).to(torch.int8)

    # Multiply with weight
    output_int8 = output_int8 * weight_int8

    return output_int8.to(torch.float32)  # Return as float32 for easier downstream use

function_signature = {
    "name": "torch_scatter_add_softsign_layer_norm_int8",
    "inputs": [
        ((16, 4), torch.float32),
        ((16,), torch.int32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((16, 4), torch.float32)
    ]
}
