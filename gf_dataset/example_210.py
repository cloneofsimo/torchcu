
import torch
import torch.nn.functional as F
from torch.fft import fft2, ifft2

class MyModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, sparsity_pattern):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.sparsity_pattern = sparsity_pattern
        self.register_buffer('mask', torch.ones_like(self.conv.weight).bool())
        self.apply_structured_sparsity()

    def apply_structured_sparsity(self):
        if self.sparsity_pattern == 'row':
            self.mask[::2, :, :, :] = False
        elif self.sparsity_pattern == 'column':
            self.mask[:, ::2, :, :] = False
        elif self.sparsity_pattern == 'checkerboard':
            self.mask[::2, ::2, :, :] = False
            self.mask[1::2, 1::2, :, :] = False

    def forward(self, x):
        self.conv.weight.data.mul_(self.mask)  # Apply sparsity pattern
        x = F.conv2d(x, self.conv.weight, None, self.conv.stride, self.conv.padding)
        x = F.relu(x)
        x = F.adaptive_max_pool3d(x.unsqueeze(1), (2, 2, 2)).squeeze(1)  # Adaptive max pooling
        return x

def torch_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a sparse convolution with FFT, followed by adaptive max pooling.
    """
    # Create an instance of the module
    module = MyModule(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, sparsity_pattern='checkerboard')

    # Apply the module
    output = module(input_tensor)

    # Return the output
    return output

function_signature = {
    "name": "torch_function",
    "inputs": [
        ((32, 3, 224, 224), torch.float32),
    ],
    "outputs": [
        ((32, 16, 56, 56), torch.float32)
    ]
}
