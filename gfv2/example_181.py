
import torch

def median_expand_detr_transformer_bf16(input_tensor: torch.Tensor, query_embed: torch.Tensor, 
                                      mask_embed: torch.Tensor) -> torch.Tensor:
    """
    Calculates median along a specific dimension, expands the result,
    applies a DETR transformer, and returns the output in bfloat16.
    """
    # Calculate median along dimension 1
    median = torch.median(input_tensor, dim=1).values
    # Expand the median to match the shape of query_embed
    median_expanded = median.unsqueeze(1).expand_as(query_embed)
    # Apply DETR transformer (assuming it's already defined)
    output = detr_transformer(query_embed, median_expanded, mask_embed)
    # Convert output to bfloat16
    return output.to(torch.bfloat16)

function_signature = {
    "name": "median_expand_detr_transformer_bf16",
    "inputs": [
        ((10, 20), torch.float32),
        ((10, 256), torch.float32),
        ((10, 10), torch.float32),
    ],
    "outputs": [
        ((10, 256), torch.bfloat16),
    ]
}
