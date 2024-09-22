
import torch
import torch.nn.functional as F

class SobelFilter(torch.nn.Module):
    def __init__(self, kernel_size=3):
        super(SobelFilter, self).__init__()
        self.kernel_size = kernel_size
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, kernel_size, kernel_size)
        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).reshape(1, 1, kernel_size, kernel_size)

    def forward(self, input):
        # Apply Sobel filter in both x and y directions
        grad_x = F.conv2d(input, self.sobel_x, padding=self.kernel_size//2)
        grad_y = F.conv2d(input, self.sobel_y, padding=self.kernel_size//2)

        # Combine the gradients
        gradient_magnitude = torch.sqrt(torch.square(grad_x) + torch.square(grad_y))
        return gradient_magnitude

def torch_sobel_filter_function(input_tensor: torch.Tensor) -> torch.Tensor:
    sobel_filter = SobelFilter()
    return sobel_filter(input_tensor)

function_signature = {
    "name": "torch_sobel_filter_function",
    "inputs": [
        ((1, 1, 10, 10), torch.float32)
    ],
    "outputs": [
        ((1, 1, 10, 10), torch.float32)
    ]
}
