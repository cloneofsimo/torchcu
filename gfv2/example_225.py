
import torch
import torch.nn.functional as F

def adaptive_max_pool_int8_function(input_tensor: torch.Tensor, output_size: int) -> torch.Tensor:
    """
    Performs adaptive max pooling on an int8 tensor, followed by a linear transformation, then returns the output.
    """
    input_tensor = input_tensor.to(torch.int8)
    output_tensor = F.adaptive_max_pool2d(input_tensor, output_size)
    output_tensor = output_tensor.to(torch.float32)
    return output_tensor

function_signature = {
    "name": "adaptive_max_pool_int8_function",
    "inputs": [
        ((16, 16, 32, 32), torch.int8),
        (16,)
    ],
    "outputs": [
        ((16, 16, 16, 16), torch.float32),
    ]
}
