
import torch

def torch_cholesky_pool_function(input_tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Applies Cholesky decomposition and average pooling to the input tensor.
    """
    # Cholesky decomposition
    L = torch.linalg.cholesky(input_tensor)
    # Average pooling
    output = torch.nn.functional.avg_pool1d(L, kernel_size=kernel_size)
    return output

function_signature = {
    "name": "torch_cholesky_pool_function",
    "inputs": [
        ((2, 3, 4), torch.float32),
        (2, )
    ],
    "outputs": [
        ((2, 3, 2), torch.float32)
    ]
}
