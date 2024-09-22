
import torch
import torch.nn.functional as F

def cross_fade_cudnn(input1: torch.Tensor, input2: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Cross-fades between two tensors using cuDNN.
    """
    return F.interpolate(input1 * alpha + input2 * (1 - alpha), size=input1.size()[2:])

function_signature = {
    "name": "cross_fade_cudnn",
    "inputs": [
        ((1, 3, 128, 128), torch.float32),
        ((1, 3, 128, 128), torch.float32),
        ((), torch.float32),
    ],
    "outputs": [
        ((1, 3, 128, 128), torch.float32),
    ]
}
