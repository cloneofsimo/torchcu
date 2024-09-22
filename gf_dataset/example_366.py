
import torch
import torch.nn.functional as F
from torch.nn import Parameter

class BilateralFilter(torch.nn.Module):
    def __init__(self, spatial_radius: int, color_radius: float, num_channels: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.spatial_radius = spatial_radius
        self.color_radius = color_radius
        self.num_channels = num_channels
        self.dtype = dtype
        self.register_buffer("weight", self.generate_weight())

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Calculate the weight tensor for each pixel
        weights = self.calculate_weights(input_tensor)

        # Apply the weights to the input tensor
        output_tensor = torch.zeros_like(input_tensor, dtype=self.dtype)
        for i in range(input_tensor.shape[0]):
            for j in range(input_tensor.shape[1]):
                for k in range(input_tensor.shape[2]):
                    output_tensor[i, j, k] = torch.sum(input_tensor[i, j, k] * weights[i, j, k]) / torch.sum(weights[i, j, k])

        return output_tensor

    def calculate_weights(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Calculate the spatial distance
        spatial_distance = torch.abs(torch.arange(2 * self.spatial_radius + 1) - self.spatial_radius)

        # Calculate the color distance
        color_distance = torch.abs(input_tensor - input_tensor.unsqueeze(1).unsqueeze(1).repeat(1, 1, 1, 2 * self.spatial_radius + 1, 2 * self.spatial_radius + 1))

        # Calculate the weights
        weights = torch.exp(-(spatial_distance ** 2 / (2 * self.spatial_radius ** 2)) - (color_distance ** 2 / (2 * self.color_radius ** 2)))
        return weights

    def generate_weight(self) -> torch.Tensor:
        # Create a weight tensor of shape (num_channels, 2 * spatial_radius + 1, 2 * spatial_radius + 1)
        return torch.ones((self.num_channels, 2 * self.spatial_radius + 1, 2 * self.spatial_radius + 1), dtype=self.dtype)

def torch_bilateral_filter_function(input_tensor: torch.Tensor, spatial_radius: int, color_radius: float, num_channels: int) -> torch.Tensor:
    """
    Applies a bilateral filter to the input tensor.
    """
    bilateral_filter = BilateralFilter(spatial_radius, color_radius, num_channels).to(input_tensor.device)
    return bilateral_filter(input_tensor)

function_signature = {
    "name": "torch_bilateral_filter_function",
    "inputs": [
        ((10, 3, 100, 100), torch.float32),
        (1, torch.int32),
        (1, torch.float32),
        (1, torch.int32),
    ],
    "outputs": [
        ((10, 3, 100, 100), torch.float32),
    ]
}
