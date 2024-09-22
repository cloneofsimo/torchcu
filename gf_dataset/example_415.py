
import torch
import torch.nn.functional as F

def torch_scaled_dot_product_attention_fp16(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Performs scaled dot-product attention with FP16 precision.
    """
    query = query.to(torch.float16)
    key = key.to(torch.float16)
    value = value.to(torch.float16)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(query.shape[-1], dtype=torch.float16))

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)

    # Compute weighted sum of values
    output = torch.matmul(attention_weights, value)
    return output.to(torch.float32)  # Return in FP32

function_signature = {
    "name": "torch_scaled_dot_product_attention_fp16",
    "inputs": [
        ((8, 16, 32), torch.float32),
        ((8, 16, 32), torch.float32),
        ((8, 16, 32), torch.float32),
        ((8, 16), torch.bool)
    ],
    "outputs": [
        ((8, 16, 32), torch.float32)
    ]
}
