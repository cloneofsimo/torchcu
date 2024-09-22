
import torch
import torch.nn.functional as F
from torch.cuda import amp

def torch_hinge_embedding_loss_bf16(input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the hinge embedding loss with bfloat16 precision.
    """
    with amp.autocast(dtype=torch.bfloat16):
        # Adaptive average pooling for each input
        input1 = F.adaptive_avg_pool2d(input1, (1, 1))
        input2 = F.adaptive_avg_pool2d(input2, (1, 1))
        # Calculate the hinge embedding loss
        loss = F.hinge_embedding_loss(input1, input2, target)
    return loss.to(torch.float32)

function_signature = {
    "name": "torch_hinge_embedding_loss_bf16",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),
        ((1, 3, 224, 224), torch.float32),
        ((1,), torch.int64)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
