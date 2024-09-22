
import torch
import torch.nn.functional as F

def torch_image_enhancement(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs image enhancement using a combination of sharpening and median filtering.
    """
    # Sharpening using Roberts cross gradient
    grad_x = torch.abs(F.conv2d(input_tensor, torch.tensor([[-1, 0], [0, 1]], dtype=torch.float32), padding=1))
    grad_y = torch.abs(F.conv2d(input_tensor, torch.tensor([[0, -1], [1, 0]], dtype=torch.float32), padding=1))
    sharpened = input_tensor + 0.5 * (grad_x + grad_y)

    # Median filtering for noise reduction
    enhanced = F.median_blur(sharpened, kernel_size=3)
    
    return enhanced.to(torch.float16)

function_signature = {
    "name": "torch_image_enhancement",
    "inputs": [
        ((3, 224, 224), torch.float32)
    ],
    "outputs": [
        ((3, 224, 224), torch.float16)
    ]
}
