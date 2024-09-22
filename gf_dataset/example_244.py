
import torch
import torch.nn.functional as F

def torch_bmm_focal_loss_fp16(input_tensor: torch.Tensor, target_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Performs a batch matrix multiplication followed by sigmoid focal loss calculation in fp16.

    Args:
        input_tensor: Input tensor of shape (batch_size, num_classes, num_features)
        target_tensor: Target tensor of shape (batch_size, num_classes)
        weights: Weights tensor of shape (batch_size, num_classes)

    Returns:
        A tensor of shape (batch_size,) representing the focal loss for each batch element.
    """
    input_fp16 = input_tensor.to(torch.float16)
    target_fp16 = target_tensor.to(torch.float16)
    weights_fp16 = weights.to(torch.float16)

    output = torch.bmm(input_fp16, target_fp16.unsqueeze(2))
    output_fp16 = output.squeeze(2).to(torch.float16)
    loss = F.binary_cross_entropy_with_logits(output_fp16, target_fp16, weights=weights_fp16, reduction='none')
    focal_loss = (1 - output_fp16).pow(2) * loss
    return focal_loss.sum(dim=1).to(torch.float32)

function_signature = {
    "name": "torch_bmm_focal_loss_fp16",
    "inputs": [
        ((16, 10, 10), torch.float32),
        ((16, 10), torch.float32),
        ((16, 10), torch.float32)
    ],
    "outputs": [
        ((16,), torch.float32)
    ]
}
