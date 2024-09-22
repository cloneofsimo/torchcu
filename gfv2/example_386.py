
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, weight):
        # Input tensor is of size (B, C, H, W)
        B, C, H, W = input_tensor.size()
        
        # Apply the module to the input
        output = MyModule(C, C)(input_tensor)

        # Resize the output to match the input size
        output = F.interpolate(output, size=(H, W), mode='bicubic', align_corners=False)

        # Generate a grid for warping
        theta = torch.randn(B, 2, 3, device=input_tensor.device)
        grid = F.affine_grid(theta, input_tensor.size())

        # Warp the output using the generated grid
        warped_output = F.grid_sample(output, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        # Apply exponential function and thresholding
        warped_output = torch.exp(warped_output)
        warped_output = torch.where(warped_output > 1.0, warped_output, torch.ones_like(warped_output))

        # Convert to int8 for memory efficiency
        warped_output = warped_output.to(torch.int8)

        # Store the intermediate results for backward pass
        ctx.save_for_backward(input_tensor, weight, warped_output, grid)

        # Return the output and the grid as a list
        return warped_output, grid

    @staticmethod
    def backward(ctx, grad_output, grad_grid):
        input_tensor, weight, warped_output, grid = ctx.saved_tensors

        # Backward pass for grid_sample
        grad_output = grad_output.to(torch.float16)
        grad_input = F.grid_sample(grad_output, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        # Backward pass for exponential function
        grad_input = grad_input * torch.where(warped_output > 1.0, warped_output, torch.zeros_like(warped_output))

        # Backward pass for the module
        grad_input = MyModule(C, C).backward(input_tensor, grad_input)

        return grad_input, None, None, None, None

function_signature = {
    "name": "my_function",
    "inputs": [
        ((4, 4, 16, 16), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4, 16, 16), torch.int8),
        ((4, 2, 3), torch.float32)
    ]
}
