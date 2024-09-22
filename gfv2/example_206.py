
import torch

def clipped_bfloat16_matmul(input_tensor: torch.Tensor, weight: torch.Tensor, clip_value: float = 1.0) -> torch.Tensor:
    """
    Performs a matrix multiplication using bfloat16, applies ReLU activation, and clips the gradient of the output.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    output = torch.matmul(input_bf16, weight_bf16.t())
    output = torch.relu(output)
    output.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
    return output.to(torch.float32)

function_signature = {
    "name": "clipped_bfloat16_matmul",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
