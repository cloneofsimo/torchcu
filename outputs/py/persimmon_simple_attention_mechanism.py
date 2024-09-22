import torch
import math

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    This function implements the scaled dot product attention mechanism.
    
    Args:
    query (torch.Tensor): The query tensor.
    key (torch.Tensor): The key tensor.
    value (torch.Tensor): The value tensor.
    
    Returns:
    torch.Tensor: The output tensor.
    """
    
    # Get the dimensions of the query, key, and value tensors
    query_dim = query.shape[-1]
    key_dim = key.shape[-1]
    value_dim = value.shape[-1]
    
    # Check if the dimensions are compatible
    assert query_dim == key_dim, "Query and key dimensions must be the same"
    assert key_dim == value_dim, "Key and value dimensions must be the same"
    
    # Calculate the dot product of the query and key tensors
    dot_product = torch.matmul(query, key.T)
    
    # Calculate the scaling factor
    scaling_factor = 1 / math.sqrt(query_dim)
    
    # Scale the dot product
    scaled_dot_product = dot_product * scaling_factor
    
    # Apply the softmax function to the scaled dot product
    attention_weights = torch.softmax(scaled_dot_product, dim=-1)
    
    # Calculate the output by taking the dot product of the attention weights and the value tensor
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