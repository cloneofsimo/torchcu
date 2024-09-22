import torch

def compute_attention_weights(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the attention weights using the query, key, and value tensors.

    Args:
        query (torch.Tensor): The query tensor.
        key (torch.Tensor): The key tensor.
        value (torch.Tensor): The value tensor.
        attention_mask (torch.Tensor): The attention mask tensor.

    Returns:
        torch.Tensor: The attention weights tensor.
    """
    attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (key.shape[-1] ** 0.5)
    attention_scores = attention_scores + attention_mask.unsqueeze(-1)
    attention_weights = torch.softmax(attention_scores, dim=-1)
    return attention_weights



# function_signature
function_signature = {
    "name": "compute_attention_weights",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [((4, 4, 4), torch.float32)]
}