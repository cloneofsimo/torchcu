
import torch

def self_attention_int8(input_tensor: torch.Tensor, query_weight: torch.Tensor, key_weight: torch.Tensor, value_weight: torch.Tensor) -> torch.Tensor:
    """
    Performs self-attention with int8 quantization.
    """
    # Quantize inputs to int8
    input_int8 = input_tensor.to(torch.int8)
    query_int8 = query_weight.to(torch.int8)
    key_int8 = key_weight.to(torch.int8)
    value_int8 = value_weight.to(torch.int8)

    # Perform self-attention in int8
    query = torch.matmul(input_int8, query_int8.t())
    key = torch.matmul(input_int8, key_int8.t())
    value = torch.matmul(input_int8, value_int8.t())
    attention_scores = torch.softmax(query / (key.shape[-1] ** 0.5), dim=-1)
    output_int8 = torch.matmul(attention_scores, value)

    # Dequantize output to float32
    output = output_int8.to(torch.float32)
    return output

function_signature = {
    "name": "self_attention_int8",
    "inputs": [
        ((16, 10, 10), torch.float32),
        ((10, 10), torch.float32),
        ((10, 10), torch.float32),
        ((10, 10), torch.float32)
    ],
    "outputs": [
        ((16, 10, 10), torch.float32),
    ]
}
