
import torch

def torch_image_processing_function(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Applies a series of image processing operations using bfloat16 precision.
    """
    # Convert to bfloat16
    image_bf16 = image.to(torch.bfloat16)

    # Calculate pairwise distances
    distances = torch.cdist(image_bf16, image_bf16)

    # Floor the distances
    floored_distances = torch.floor(distances)

    # Roberts Cross Gradient
    gradient_x = torch.diff(image_bf16, dim=1)
    gradient_y = torch.diff(image_bf16, dim=0)
    roberts_gradient = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Frobenius norm of the gradient
    frobenius_norm = torch.linalg.norm(roberts_gradient, ord='fro')

    # Combine results and return
    output = torch.stack((floored_distances, roberts_gradient, frobenius_norm), dim=0)
    return output.to(torch.float32)

function_signature = {
    "name": "torch_image_processing_function",
    "inputs": [
        ((32, 32, 3), torch.float32),
        (1, torch.int32)
    ],
    "outputs": [
        ((32, 32, 32), torch.float32),
        ((32, 32, 3), torch.float32),
        ((), torch.float32)
    ]
}
