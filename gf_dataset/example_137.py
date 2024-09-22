
import torch
import torch.nn.functional as F

def torch_segmentation_function(image: torch.Tensor, markers: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Performs watershed segmentation on an image with given markers and weights.
    """

    # Adaptive average pooling
    image_pooled = F.adaptive_avg_pool3d(image, (1, 1, 1))

    # Reshape for watershed
    image_pooled = image_pooled.reshape(image_pooled.shape[0], -1)
    markers = markers.reshape(markers.shape[0], -1)

    # Watershed segmentation (using an external library for this example)
    # Replace this with your preferred watershed implementation
    from skimage.segmentation import watershed
    segmented_image = watershed(image_pooled.cpu().numpy(), markers.cpu().numpy(), weights=weights.cpu().numpy())

    # Grid sampling with weights
    grid = torch.arange(0, segmented_image.shape[1], dtype=torch.float32).view(1, 1, -1).expand(segmented_image.shape[0], 1, -1) / (segmented_image.shape[1] - 1)
    segmented_image = torch.from_numpy(segmented_image).to(image.device)
    segmented_image = segmented_image.unsqueeze(1)
    sampled_image = F.grid_sample(image, grid.to(image.device), mode='bilinear', align_corners=True).squeeze(1)

    # Re-shape for original size
    sampled_image = sampled_image.reshape(image.shape)

    # Apply weights
    weighted_image = sampled_image * weights

    return weighted_image

function_signature = {
    "name": "torch_segmentation_function",
    "inputs": [
        ((1, 3, 128, 128, 128), torch.float32),
        ((1, 1, 128, 128, 128), torch.int32),
        ((1, 1, 128, 128, 128), torch.float32),
    ],
    "outputs": [
        ((1, 3, 128, 128, 128), torch.float32),
    ]
}
