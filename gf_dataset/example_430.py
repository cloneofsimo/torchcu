
import torch
import torch.nn.functional as F

def torch_image_enhancement(input_tensor: torch.Tensor, weight: torch.Tensor, label: torch.Tensor, smoothing_factor: float = 0.1) -> torch.Tensor:
    """
    Performs image enhancement using a combination of:
        1. Linear transformation with bfloat16 precision
        2. Pixel shuffling for upsampling
        3. Label smoothing for regularization
        4. Addmm for weighted combination with label

    Args:
        input_tensor (torch.Tensor): Input image tensor (B, C, H, W).
        weight (torch.Tensor): Weight tensor for linear transformation (C, C).
        label (torch.Tensor): Label tensor for weighted combination (B, C, H, W).
        smoothing_factor (float): Label smoothing factor. Defaults to 0.1.

    Returns:
        torch.Tensor: Enhanced image tensor (B, C, H, W).
    """
    # 1. Linear transformation with bfloat16
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    output_bf16 = torch.bmm(input_bf16.view(input_tensor.shape[0], input_tensor.shape[1], -1), weight_bf16.t()).view(input_tensor.shape)

    # 2. Pixel shuffling for upsampling
    output_upsampled = F.pixel_shuffle(output_bf16, upscale_factor=2)

    # 3. Label smoothing for regularization
    label_smoothed = (1 - smoothing_factor) * label + smoothing_factor * torch.ones_like(label) / label.shape[1]

    # 4. Addmm for weighted combination with label
    output = torch.baddbmm(label_smoothed, input_tensor.view(input_tensor.shape[0], 1, -1), weight.t()).view(input_tensor.shape)

    return output.clamp(0, 1)

function_signature = {
    "name": "torch_image_enhancement",
    "inputs": [
        ((1, 3, 32, 32), torch.float32),
        ((3, 3), torch.float32),
        ((1, 3, 32, 32), torch.float32),
        (0.1, torch.float32)
    ],
    "outputs": [
        ((1, 3, 64, 64), torch.float32)
    ]
}
