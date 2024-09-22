import torch
import math

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    Compute 'Scaled Dot Product Attention'
    """
    # Get the dimensions of the query, key and value tensors
    query_dim = query.size(-1)
    key_dim = key.size(-1)
    value_dim = value.size(-1)

    # Check if the dimensions are compatible
    assert query_dim == key_dim, "Query and key dimensions must be the same"
    assert key_dim == value_dim, "Key and value dimensions must be the same"

    # Compute the attention scores
    attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query_dim)

    # Compute the attention weights
    attention_weights = torch.softmax(attention_scores, dim=-1)

    # Compute the output
    output = torch.matmul(attention_weights, value)

    return output



# function_signature
function_signature = {
    "name": "scaled_dot_product_attention",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [((4, 4), torch.float32)]
}