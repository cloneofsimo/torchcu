
import torch
import torch.nn.functional as F

def mish_time_stretch_causal_attention_int8(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                           stretch_factor: int, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Applies Mish activation, time stretching, causal attention, and int8 quantization to the input tensors.
    
    Args:
        query (torch.Tensor): Query tensor with shape (batch_size, query_len, hidden_dim).
        key (torch.Tensor): Key tensor with shape (batch_size, key_len, hidden_dim).
        value (torch.Tensor): Value tensor with shape (batch_size, value_len, hidden_dim).
        stretch_factor (int): Factor to stretch the time dimension of query and key tensors.
        mask (torch.Tensor, optional): Optional mask tensor with shape (batch_size, query_len, key_len).
                                          Defaults to None.

    Returns:
        torch.Tensor: Output tensor with shape (batch_size, query_len, hidden_dim).
    """
    
    # Mish activation
    query = F.mish(query)
    key = F.mish(key)
    
    # Time stretching (repeatedly stack the tensors)
    query = query.repeat_interleave(stretch_factor, dim=1)
    key = key.repeat_interleave(stretch_factor, dim=1)
    
    # Causal attention
    attention = torch.matmul(query, key.transpose(-2, -1))
    if mask is not None:
        attention = attention.masked_fill(mask == 0, float('-inf'))
    attention = F.softmax(attention, dim=-1)
    output = torch.matmul(attention, value)
    
    # Int8 quantization
    output = output.to(torch.int8)
    
    return output

function_signature = {
    "name": "mish_time_stretch_causal_attention_int8",
    "inputs": [
        ((1, 10, 32), torch.float32),
        ((1, 10, 32), torch.float32),
        ((1, 20, 32), torch.float32),
        (1, ), torch.int32,
    ],
    "outputs": [
        ((1, 10, 32), torch.int8),
    ]
}
