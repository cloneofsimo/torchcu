
import torch

def torch_int8_crossfade_dot(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor, weight: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Computes the dot product of two int8 tensors, applies a cross-fade with a given alpha, and returns the result.
    """
    input_tensor1_int8 = input_tensor1.to(torch.int8)
    input_tensor2_int8 = input_tensor2.to(torch.int8)
    weight_int8 = weight.to(torch.int8)

    dot_product = torch.dot(input_tensor1_int8, input_tensor2_int8)
    cross_fade = (1 - alpha) * dot_product + alpha * torch.sum(weight_int8)
    return cross_fade.to(torch.float32)

function_signature = {
    "name": "torch_int8_crossfade_dot",
    "inputs": [
        ((10,), torch.float32),
        ((10,), torch.float32),
        ((10,), torch.float32),
        ((), torch.float32)
    ],
    "outputs": [
        ((), torch.float32),
    ]
}
