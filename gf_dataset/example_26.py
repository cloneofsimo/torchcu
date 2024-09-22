
import torch
import torch.nn as nn

class CannyEdgeDetector(nn.Module):
    def __init__(self, low_threshold: float = 0.1, high_threshold: float = 0.2):
        super().__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Apply Gaussian blur
        blurred_image = torch.nn.functional.gaussian_blur(image, kernel_size=5, sigma=1)

        # Compute gradients
        gradient_x = torch.nn.functional.conv2d(blurred_image, torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float(), padding=1)
        gradient_y = torch.nn.functional.conv2d(blurred_image, torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float(), padding=1)

        # Calculate gradient magnitude and direction
        gradient_magnitude = torch.sqrt(torch.square(gradient_x) + torch.square(gradient_y))
        gradient_direction = torch.atan2(gradient_y, gradient_x)

        # Apply non-maximum suppression
        nms_image = self.non_maximum_suppression(gradient_magnitude, gradient_direction)

        # Apply hysteresis thresholding
        edges = self.hysteresis_thresholding(nms_image, self.low_threshold, self.high_threshold)

        return edges

    def non_maximum_suppression(self, magnitude: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """Performs non-maximum suppression to thin out the edges."""
        # ... (implementation omitted for brevity) ...

    def hysteresis_thresholding(self, nms_image: torch.Tensor, low_threshold: float, high_threshold: float) -> torch.Tensor:
        """Applies hysteresis thresholding to connect edges."""
        # ... (implementation omitted for brevity) ...

def torch_canny_edge_detection_function(image: torch.Tensor, low_threshold: float = 0.1, high_threshold: float = 0.2) -> torch.Tensor:
    """
    Performs Canny edge detection on an image.
    """
    canny_detector = CannyEdgeDetector(low_threshold, high_threshold)
    edges = canny_detector(image)
    return edges

function_signature = {
    "name": "torch_canny_edge_detection_function",
    "inputs": [
        ((1, 1, 256, 256), torch.float32),
        (torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((1, 1, 256, 256), torch.float32),
    ]
}
