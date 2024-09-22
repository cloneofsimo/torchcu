
import torch
import torch.nn.functional as F

def mish_global_attention_mul(input_tensor: torch.Tensor, attention_weights: torch.Tensor, 
                              scale: float = 1.0) -> torch.Tensor:
    """
    Applies Mish activation, global attention, and element-wise multiplication.

    Args:
        input_tensor: Input tensor of shape (batch_size, seq_len, hidden_dim).
        attention_weights: Attention weights of shape (batch_size, seq_len).
        scale: Scaling factor for the element-wise multiplication.

    Returns:
        Output tensor of the same shape as input_tensor.
    """
    mish_output = F.mish(input_tensor)
    
    # Reshape attention weights for broadcasting
    attention_weights = attention_weights.unsqueeze(-1)

    # Apply global attention
    attended_output = mish_output * attention_weights

    # Element-wise multiplication
    return attended_output * scale

function_signature = {
    "name": "mish_global_attention_mul",
    "inputs": [
        ((1, 16, 128), torch.float32),
        ((1, 16), torch.float32),
        (1, torch.float32)
    ],
    "outputs": [
        ((1, 16, 128), torch.float32),
    ]
}
