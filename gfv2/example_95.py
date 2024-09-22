
import torch

def binary_cross_entropy_fp16(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates binary cross-entropy loss between input and target tensors using fp16.
    """
    input_fp16 = input_tensor.to(torch.float16)
    target_fp16 = target_tensor.to(torch.float16)
    loss = torch.nn.functional.binary_cross_entropy(input_fp16, target_fp16)
    return loss.to(torch.float32)

function_signature = {
    "name": "binary_cross_entropy_fp16",
    "inputs": [
        ((10, 10), torch.float32),
        ((10, 10), torch.float32),
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
