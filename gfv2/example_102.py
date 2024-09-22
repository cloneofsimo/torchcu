
import torch

def adaptive_avg_pool_function(input_tensor: torch.Tensor, output_size: int) -> torch.Tensor:
    """
    Performs adaptive average pooling on the input tensor.
    """
    output = torch.nn.functional.adaptive_avg_pool2d(input_tensor, output_size)
    return output

function_signature = {
    "name": "adaptive_avg_pool_function",
    "inputs": [
        ((3, 10, 10), torch.float32),
        (1, torch.int32)
    ],
    "outputs": [
        ((3, 1, 1), torch.float32)
    ]
}
