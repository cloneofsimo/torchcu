
import torch
import torch.nn.functional as F

def gaussian_maxpool_function(input_tensor: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """
    Applies a Gaussian blur to the input tensor followed by a 1D max pooling operation.
    """
    # Gaussian blur
    input_tensor = F.gaussian_blur(input_tensor, kernel_size=kernel_size, sigma=1.0)

    # 1D max pooling with specified stride
    output_tensor = F.max_pool1d(input_tensor, kernel_size=kernel_size, stride=stride)
    return output_tensor


function_signature = {
    "name": "gaussian_maxpool_function",
    "inputs": [
        ((1, 16, 32), torch.float32),
        (3, torch.int32),
        (2, torch.int32),
    ],
    "outputs": [
        ((1, 16, 16), torch.float32),
    ]
}
