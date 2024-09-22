
import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelFilter(nn.Module):
    def __init__(self, in_channels=1):
        super(SobelFilter, self).__init__()
        self.kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        self.kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.kernel_x = self.kernel_x.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        self.kernel_y = self.kernel_y.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)

    def forward(self, x):
        x_grad = F.conv2d(x, self.kernel_x, padding=1)
        y_grad = F.conv2d(x, self.kernel_y, padding=1)
        grad_magnitude = torch.sqrt(torch.square(x_grad) + torch.square(y_grad))
        return grad_magnitude.int()

def torch_sobel_filter_int8_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies Sobel filter to input tensor and returns the gradient magnitude in int8.
    """
    torch.manual_seed(42)
    sobel = SobelFilter(in_channels=input_tensor.shape[1])
    output = sobel(input_tensor)
    return output

function_signature = {
    "name": "torch_sobel_filter_int8_function",
    "inputs": [
        ((1, 1, 10, 10), torch.float32),
    ],
    "outputs": [
        ((1, 1, 10, 10), torch.int8),
    ]
}
