
import torch
import torch.nn.functional as F

def multi_scale_attention_fp16(input_tensor: torch.Tensor, query_tensor: torch.Tensor, key_tensor: torch.Tensor, 
                               value_tensor: torch.Tensor, scales: list) -> torch.Tensor:
    """
    Performs multi-scale attention with fp16 precision.

    Args:
        input_tensor: Input tensor (B, T, D)
        query_tensor: Query tensor (B, T, D)
        key_tensor: Key tensor (B, T, D)
        value_tensor: Value tensor (B, T, D)
        scales: List of attention scales (e.g., [1, 2, 4])

    Returns:
        Output tensor (B, T, D)
    """
    input_tensor = input_tensor.to(torch.float16)
    query_tensor = query_tensor.to(torch.float16)
    key_tensor = key_tensor.to(torch.float16)
    value_tensor = value_tensor.to(torch.float16)

    output = torch.zeros_like(input_tensor, dtype=torch.float16)

    for scale in scales:
        # Transpose key and value tensors for efficient attention calculation
        key_transposed = key_tensor.transpose(1, 2)
        value_transposed = value_tensor.transpose(1, 2)

        # Calculate attention scores
        attention_scores = torch.matmul(query_tensor, key_transposed)

        # Apply scaling to attention scores
        attention_scores = attention_scores / (query_tensor.shape[-1] ** 0.5 * scale)

        # Apply softmax to obtain attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Calculate context vector
        context = torch.matmul(attention_weights, value_transposed)

        # Transpose back for the next iteration
        context = context.transpose(1, 2)

        # Add to the final output
        output += context

    return output.to(torch.float32)

function_signature = {
    "name": "multi_scale_attention_fp16",
    "inputs": [
        ((16, 1024, 512), torch.float32),
        ((16, 1024, 512), torch.float32),
        ((16, 1024, 512), torch.float32),
        ((16, 1024, 512), torch.float32),
        ([1, 2, 4], torch.int32)
    ],
    "outputs": [
        ((16, 1024, 512), torch.float32)
    ]
}
