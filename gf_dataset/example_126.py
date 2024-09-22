
import torch
import torch.nn.functional as F

def torch_l1_loss_pooling_bf16(input_tensor: torch.Tensor, target_tensor: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Calculates L1 loss between input and target tensors after applying adaptive max pooling.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    target_bf16 = target_tensor.to(torch.bfloat16)
    pooled_input = F.adaptive_max_pool3d(input_bf16, (kernel_size, kernel_size, kernel_size))
    pooled_target = F.adaptive_max_pool3d(target_bf16, (kernel_size, kernel_size, kernel_size))
    l1_loss = F.l1_loss(pooled_input, pooled_target, reduction='mean')
    return l1_loss.to(torch.float32)

function_signature = {
    "name": "torch_l1_loss_pooling_bf16",
    "inputs": [
        ((1, 4, 8, 8, 8), torch.float32),
        ((1, 4, 8, 8, 8), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
