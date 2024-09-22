
import torch
import torch.nn.functional as F

def causal_attention_int8_gradient_magnitude(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Performs causal attention with int8 quantization, computes gradients and returns their magnitude.
    """
    query_int8 = query.to(torch.int8)
    key_int8 = key.to(torch.int8)
    value_int8 = value.to(torch.int8)

    # Perform causal attention using int8 tensors
    output = F.scaled_dot_product_attention(query_int8, key_int8, value_int8, attn_mask=mask, dropout=0.0)

    # Calculate gradients
    output.backward()

    # Return gradient magnitude of the query tensor
    return torch.linalg.norm(query.grad)

function_signature = {
    "name": "causal_attention_int8_gradient_magnitude",
    "inputs": [
        ((4, 4, 8), torch.float32),
        ((4, 4, 8), torch.float32),
        ((4, 4, 8), torch.float32),
        ((4, 4), torch.bool)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
