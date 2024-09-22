
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

def torch_contrastive_dilation_function(input_tensor: torch.Tensor, kernel: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Performs morphological dilation on the input tensor, then calculates the supervised contrastive loss.

    Args:
        input_tensor: The input tensor.
        kernel: The dilation kernel.
        labels: The labels for contrastive loss.

    Returns:
        The supervised contrastive loss.
    """
    dilated_input = F.max_pool2d(input_tensor, kernel_size=kernel.shape, stride=1, padding=kernel.shape[0] // 2, dilation=kernel.shape[0] // 2)
    
    loss = supervised_contrastive_loss(dilated_input, labels)
    return loss

function_signature = {
    "name": "torch_contrastive_dilation_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((3, 3), torch.float32),
        ((4, 4), torch.long)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
