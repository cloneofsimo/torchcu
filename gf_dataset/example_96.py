
import torch
import torch.nn.functional as F

def torch_image_filter(input_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Applies a 2D convolution with the given kernel using reflection padding.
    """
    padded_input = F.pad(input_tensor, (kernel.shape[1]//2, kernel.shape[1]//2, kernel.shape[2]//2, kernel.shape[2]//2), 'reflect')
    output = F.conv2d(padded_input, kernel, groups=input_tensor.shape[1])
    return output

function_signature = {
    "name": "torch_image_filter",
    "inputs": [
        ((1, 3, 224, 224), torch.float32),
        ((3, 3, 3), torch.float32)
    ],
    "outputs": [
        ((1, 3, 224, 224), torch.float32),
    ]
}
