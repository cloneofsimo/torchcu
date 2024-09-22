
import torch
import torch.nn.functional as F

def torch_scharr_upsample_fp16(input_tensor: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """
    Calculates the Scharr gradient of an image, upsamples it, and returns the result in fp16.
    """
    # Calculate Scharr gradient
    gradient_x = F.conv2d(input_tensor.to(torch.float32), torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32).reshape(1, 1, 3, 3), padding=1)
    gradient_y = F.conv2d(input_tensor.to(torch.float32), torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32).reshape(1, 1, 3, 3), padding=1)

    # Combine gradients (optional, can be returned separately)
    gradient = torch.sqrt(gradient_x**2 + gradient_y**2)

    # Upsample using bilinear interpolation
    upsampled_gradient = F.interpolate(gradient, scale_factor=scale_factor, mode='bilinear', align_corners=False)

    return upsampled_gradient.to(torch.float16)

function_signature = {
    "name": "torch_scharr_upsample_fp16",
    "inputs": [
        ((1, 1, 10, 10), torch.float32),
        (torch.float32,)
    ],
    "outputs": [
        ((1, 1, 20, 20), torch.float16)
    ]
}
