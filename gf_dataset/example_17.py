
import torch
import torch.nn.functional as F
from skimage.morphology import disk

def torch_image_laplacian_erosion_function(input_tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Calculates the Laplacian of an image, subtracts it from the original image,
    and then performs morphological erosion.
    """
    # Calculate Laplacian
    laplacian = torch.abs(torch.sum(torch.tensor([[-1, -1, -1],
                                                    [-1, 8, -1],
                                                    [-1, -1, -1]]).float().to(input_tensor.device).unsqueeze(0).unsqueeze(0) * F.conv2d(input_tensor, torch.ones(1, 1, kernel_size, kernel_size), padding=kernel_size//2), dim=1, keepdim=True))
    # Subtract Laplacian from the original image
    subtracted = input_tensor - laplacian
    # Perform erosion
    selem = disk(kernel_size)
    eroded = torch.from_numpy(morphology.binary_erosion(subtracted.cpu().numpy(), selem)).to(subtracted.dtype).to(subtracted.device)
    return eroded.float()

function_signature = {
    "name": "torch_image_laplacian_erosion_function",
    "inputs": [
        ((1, 1, 256, 256), torch.float32),
        ((), torch.int32)
    ],
    "outputs": [
        ((1, 1, 256, 256), torch.float32)
    ]
}
