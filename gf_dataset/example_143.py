
import torch

def torch_pool_unfold_inplace(input_tensor: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """
    Performs a 1D max pooling operation followed by an unfold operation inplace. 
    """
    output = torch.nn.functional.max_pool1d(input_tensor, kernel_size=kernel_size, stride=stride)
    output = torch.nn.functional.unfold(output, kernel_size=kernel_size, stride=stride)
    output.data = output.data.clamp(min=0)  # inplace operation
    return output

function_signature = {
    "name": "torch_pool_unfold_inplace",
    "inputs": [
        ((4, 4), torch.float32),
        (1, torch.int32),
        (1, torch.int32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
