
import torch

def image_gradient_fp16(image: torch.Tensor) -> torch.Tensor:
    """
    Calculates the gradient of an image using finite differences in FP16.
    """
    image_fp16 = image.to(torch.float16)
    # Calculate horizontal gradient
    gradient_x = image_fp16[:, :, 1:] - image_fp16[:, :, :-1]
    # Calculate vertical gradient
    gradient_y = image_fp16[:, 1:, :] - image_fp16[:, :-1, :]
    # Combine gradients into a single tensor
    gradient = torch.cat((gradient_x, gradient_y), dim=0)
    return gradient.to(torch.float32)

function_signature = {
    "name": "image_gradient_fp16",
    "inputs": [
        ((3, 224, 224), torch.float32)
    ],
    "outputs": [
        ((6, 223, 223), torch.float32)
    ]
}
