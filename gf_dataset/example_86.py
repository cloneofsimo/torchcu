
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm2d

def torch_batchnorm_erosion_int8_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Performs a batch normalization followed by morphological erosion on an int8 tensor.
    """
    # Batch normalization
    input_tensor = input_tensor.to(torch.int8)
    bn = BatchNorm2d(num_features=input_tensor.shape[1], weight=weight, bias=bias, running_mean=running_mean, running_var=running_var)
    output = bn(input_tensor)

    # Morphological erosion
    output = F.max_pool2d(output, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    return output.to(torch.float32)

function_signature = {
    "name": "torch_batchnorm_erosion_int8_function",
    "inputs": [
        ((1, 3, 10, 10), torch.int8),
        ((3,), torch.float32),
        ((3,), torch.float32),
        ((3,), torch.float32),
        ((3,), torch.float32),
        (3,)
    ],
    "outputs": [
        ((1, 3, 10, 10), torch.float32)
    ]
}
