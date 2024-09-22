
import torch
import torch.nn.functional as F

def torch_laplacian_enhancement(image: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Enhances an image using Laplacian filtering and Hadamard product.
    """
    image_bf16 = image.to(torch.bfloat16)
    laplacian = F.laplacian(image_bf16, kernel_size=3)
    enhanced = image_bf16 + strength * laplacian
    return enhanced.pow(2).to(torch.float32)

function_signature = {
    "name": "torch_laplacian_enhancement",
    "inputs": [
        ((3, 224, 224), torch.float32),
        ((1,), torch.float32)  # Strength as a scalar tensor
    ],
    "outputs": [
        ((3, 224, 224), torch.float32),
    ]
}
