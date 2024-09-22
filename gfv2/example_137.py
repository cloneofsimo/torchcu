
import torch
import torch.nn.functional as F
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

def watershed_segmentation_with_inner_product(image_tensor: torch.Tensor, markers_tensor: torch.Tensor, weights_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs watershed segmentation using peak local maxima as markers and applies an inner product with given weights.

    Args:
        image_tensor (torch.Tensor): Input image tensor, expected to be in the range [0, 1].
        markers_tensor (torch.Tensor): Tensor of markers for watershed segmentation.
        weights_tensor (torch.Tensor): Weights for the inner product.

    Returns:
        torch.Tensor: Segmented image tensor with inner product applied.
    """
    image = image_tensor.cpu().numpy()
    markers = markers_tensor.cpu().numpy()
    weights = weights_tensor.cpu().numpy()

    # Perform watershed segmentation
    segmentation = watershed(image, markers, mask=image > 0)

    # Apply inner product with weights
    segmented_image = (segmentation * weights).astype(np.float32)

    return torch.from_numpy(segmented_image).to(image_tensor.device)

function_signature = {
    "name": "watershed_segmentation_with_inner_product",
    "inputs": [
        ((1, 1, 100, 100), torch.float32),
        ((1, 1, 100, 100), torch.float32),
        ((100, 100), torch.float32)
    ],
    "outputs": [
        ((1, 1, 100, 100), torch.float32)
    ]
}
