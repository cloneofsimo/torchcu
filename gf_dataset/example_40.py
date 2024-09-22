
import torch
import torch.fft

def torch_canny_edge_detection(input_tensor: torch.Tensor, low_threshold: float, high_threshold: float) -> torch.Tensor:
    """
    Performs Canny edge detection on the input tensor.

    Args:
        input_tensor: Input tensor representing an image (expected to be in grayscale).
        low_threshold: Lower threshold for hysteresis.
        high_threshold: Higher threshold for hysteresis.

    Returns:
        A tensor representing the edge map.
    """
    edges = torch.zeros_like(input_tensor)
    edges[torch.where(torch.abs(torch.gradient(input_tensor, axis=0)) > high_threshold)] = 1.0
    edges[torch.where(torch.abs(torch.gradient(input_tensor, axis=1)) > high_threshold)] = 1.0

    # Perform hysteresis thresholding
    for y in range(1, input_tensor.shape[0] - 1):
        for x in range(1, input_tensor.shape[1] - 1):
            if edges[y, x] == 1.0:
                if edges[y - 1, x] == 1.0 or edges[y + 1, x] == 1.0 or \
                   edges[y, x - 1] == 1.0 or edges[y, x + 1] == 1.0:
                    edges[y, x] = 1.0
                else:
                    edges[y, x] = 0.0
    return edges

function_signature = {
    "name": "torch_canny_edge_detection",
    "inputs": [
        ((128, 128), torch.float32),
        (torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((128, 128), torch.float32),
    ]
}
