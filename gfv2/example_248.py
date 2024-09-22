
import torch
import torch.nn.functional as F

def round_local_attention_mean_int8(input_tensor: torch.Tensor, query_tensor: torch.Tensor, key_tensor: torch.Tensor, value_tensor: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Performs a local attention mechanism with int8 quantization, rounding, and mean pooling.
    """
    # Convert to int8
    input_int8 = input_tensor.to(torch.int8)
    query_int8 = query_tensor.to(torch.int8)
    key_int8 = key_tensor.to(torch.int8)
    value_int8 = value_tensor.to(torch.int8)

    # Local attention
    attention_scores = F.scaled_dot_product_attention(query_int8, key_int8, value_int8, attn_mask=None, dropout_p=0.0, is_causal=False)

    # Round to nearest integer
    rounded_attention = torch.round(attention_scores)

    # Mean pooling
    output = torch.mean(rounded_attention, dim=1)

    # Return in float32
    return output.to(torch.float32)

function_signature = {
    "name": "round_local_attention_mean_int8",
    "inputs": [
        ((16, 128), torch.float32),
        ((16, 128), torch.float32),
        ((16, 128), torch.float32),
        ((16, 128), torch.float32),
        (16, )
    ],
    "outputs": [
        ((16,), torch.float32),
    ]
}
