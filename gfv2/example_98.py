
import torch
import torch.nn.functional as F

def coord_attention_fp16_int8(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Applies a coordinate attention mechanism with fp16 precision and returns the result in int8.
    """
    input_fp16 = input_tensor.to(torch.float16)
    weight_fp16 = weight.to(torch.float16)

    # Calculate attention weights
    attention = F.softmax(torch.matmul(input_fp16, weight_fp16.t()), dim=-1)

    # Apply attention to the input
    output = torch.matmul(attention, input_fp16)

    return output.to(torch.int8)

function_signature = {
    "name": "coord_attention_fp16_int8",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.int8),
    ]
}
