
import torch

def elementwise_min_cudnn(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor) -> torch.Tensor:
    """
    Performs element-wise minimum operation using cuDNN.
    """
    return torch.minimum(input_tensor1, input_tensor2)

function_signature = {
    "name": "elementwise_min_cudnn",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
