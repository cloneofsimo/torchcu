import torch
import math

def calculate_attention_scores(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    attention_mask: torch.Tensor, 
    dropout_prob: float = 0.1
) -> torch.Tensor:
    """
    Calculate the attention scores and output of a self-attention layer.

    Args:
    query (torch.Tensor): The query tensor.
    key (torch.Tensor): The key tensor.
    value (torch.Tensor): The value tensor.
    attention_mask (torch.Tensor): The attention mask tensor.
    dropout_prob (float, optional): The dropout probability. Defaults to 0.1.

    Returns:
    torch.Tensor: The output of the self-attention layer.
    """
    # Calculate the attention scores
    attention_scores = torch.matmul(query, key.T) / math.sqrt(query.size(-1))
    attention_scores = attention_scores + attention_mask

    # Apply the softmax function to the attention scores
    attention_scores = torch.softmax(attention_scores, dim=-1)

    # Apply dropout to the attention scores
    attention_scores = torch.dropout(attention_scores, p=dropout_prob, train=True)

    # Calculate the output of the self-attention layer
    output = torch.matmul(attention_scores, value)

    return output



# function_signature
function_signature = {
    "name": "calculate_attention_scores",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [((4, 4), torch.float32)]
}