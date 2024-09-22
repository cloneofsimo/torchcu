
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def torch_celu_avgpool_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies CELU activation followed by average pooling.
    """
    with autocast():
        output = F.celu(input_tensor, alpha=1.0)
    return F.avg_pool2d(output, kernel_size=2)

function_signature = {
    "name": "torch_celu_avgpool_function",
    "inputs": [
        ((16, 3, 32, 32), torch.float32)
    ],
    "outputs": [
        ((16, 3, 16, 16), torch.float32)
    ]
}
