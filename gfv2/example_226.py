
import torch

def softmax_cross_entropy_fp16(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculate the softmax cross-entropy loss using fp16 for efficiency.
    """
    input_fp16 = input_tensor.to(torch.float16)
    target_fp16 = target_tensor.to(torch.long)
    softmax_output = torch.softmax(input_fp16, dim=1)
    loss = torch.nn.functional.cross_entropy(softmax_output, target_fp16, reduction='sum')
    return loss.to(torch.float32)

function_signature = {
    "name": "softmax_cross_entropy_fp16",
    "inputs": [
        ((10, 5), torch.float32),
        ((10,), torch.long)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
