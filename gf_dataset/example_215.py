
import torch
import torch.nn.functional as F

def pixel_shuffle_torch(input_tensor: torch.Tensor, upscale_factor: int) -> torch.Tensor:
    """
    Performs pixel shuffle operation in PyTorch.
    """
    batch_size, channels, height, width = input_tensor.size()
    
    # Reshape the input tensor
    input_tensor = input_tensor.view(batch_size, channels // (upscale_factor ** 2), upscale_factor, upscale_factor, height, width)
    
    # Transpose the dimensions
    input_tensor = input_tensor.permute(0, 1, 4, 2, 5, 3)
    
    # Reshape again to get the final output
    output_tensor = input_tensor.contiguous().view(batch_size, channels // (upscale_factor ** 2), height * upscale_factor, width * upscale_factor)
    
    return output_tensor

function_signature = {
    "name": "pixel_shuffle_torch",
    "inputs": [
        ((1, 12, 4, 4), torch.float32),
        (2,)
    ],
    "outputs": [
        ((1, 3, 8, 8), torch.float32)
    ]
}
