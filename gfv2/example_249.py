
import torch
import torch.nn.functional as F

def spatial_attention_gelu_fp16(input_tensor: torch.Tensor, weight: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    """
    Applies spatial attention, GELU activation, and layer scaling decay in fp16.
    """
    input_fp16 = input_tensor.to(torch.float16)
    weight_fp16 = weight.to(torch.float16)

    # Spatial attention
    attention_weights = F.softmax(torch.matmul(input_fp16, weight_fp16.t()), dim=-1)
    attended_input = torch.matmul(attention_weights, input_fp16)

    # GELU activation
    output_fp16 = F.gelu(attended_input)

    # Layer scaling decay (applied in-place)
    output_fp16.mul_(scaling_factor)

    return output_fp16.to(torch.float32)

function_signature = {
    "name": "spatial_attention_gelu_fp16",
    "inputs": [
        ((1, 1024, 28, 28), torch.float32),
        ((1024, 1024), torch.float32),
        (torch.float32,)
    ],
    "outputs": [
        ((1, 1024, 28, 28), torch.float32)
    ]
}
