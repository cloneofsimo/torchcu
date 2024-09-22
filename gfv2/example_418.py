
import torch

def median_filter(input_tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Applies a median filter to the input tensor with the specified kernel size.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    padding = kernel_size // 2
    output = torch.nn.functional.pad(input_tensor, (padding, padding, padding, padding), 'replicate')
    output = output.unfold(1, kernel_size, 1).unfold(2, kernel_size, 1)
    output = torch.median(output, dim=3).values
    output = output.median(dim=2).values

    return output

function_signature = {
    "name": "median_filter",
    "inputs": [
        ((16, 16), torch.float32),
        (1, torch.int32)
    ],
    "outputs": [
        ((16, 16), torch.float32),
    ]
}
