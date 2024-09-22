
import torch

def torch_adaptive_avg_pool1d_function(input_tensor: torch.Tensor, output_size: int) -> torch.Tensor:
    """
    Performs adaptive average pooling in 1D.
    """
    return torch.nn.functional.adaptive_avg_pool1d(input_tensor, output_size)

function_signature = {
    "name": "torch_adaptive_avg_pool1d_function",
    "inputs": [
        ((2, 3, 5), torch.float32),
        (1, torch.int32)  # Output size is an integer
    ],
    "outputs": [
        ((2, 3, 1), torch.float32),
    ]
}
