
import torch
import torch.nn as nn
import torch.nn.functional as F

def complex_loss_function(input_tensor: torch.Tensor, target_tensor: torch.Tensor, weights: torch.Tensor) -> list:
    """
    Calculates a complex loss combining NLL, standard deviation, and Kronecker product.
    """
    # 1. NLL Loss (Negative Log Likelihood)
    nll_loss = F.nll_loss(F.log_softmax(input_tensor, dim=1), target_tensor, reduction='mean')

    # 2. Standard Deviation across channels
    std_loss = torch.std(input_tensor, dim=1).mean()

    # 3. Kronecker Product (using a small sub-tensor for efficiency)
    kron_loss = torch.kron(input_tensor[:, :2, :2, :2], weights).mean()

    # 4. Max Pooling 3D
    pooled_tensor = F.max_pool3d(input_tensor, kernel_size=3, stride=2)

    return [nll_loss.to(torch.float32), std_loss.to(torch.float32), kron_loss.to(torch.float32), pooled_tensor.to(torch.float32)]

function_signature = {
    "name": "complex_loss_function",
    "inputs": [
        ((16, 10, 8, 8, 8), torch.float32),
        ((16,), torch.int64),
        ((2, 2, 2, 2), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
        ((1,), torch.float32),
        ((1,), torch.float32),
        ((16, 5, 4, 4, 4), torch.float32)
    ]
}
