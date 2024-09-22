
import torch

def torch_median_filter_conv1d(input_tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Applies a median filter followed by a 1D convolution.
    """
    output = torch.nn.functional.median_filter(input_tensor, kernel_size=kernel_size)
    output = torch.nn.functional.conv1d(output.unsqueeze(1), torch.ones(1, 1, kernel_size), padding=kernel_size//2).squeeze(1)
    return output

function_signature = {
    "name": "torch_median_filter_conv1d",
    "inputs": [
        ((10, 20), torch.float32),
        (3,)
    ],
    "outputs": [
        ((10, 20), torch.float32)
    ]
}
