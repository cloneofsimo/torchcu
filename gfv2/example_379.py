
import torch
import torch.nn.functional as F

def my_complex_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on the input tensor:
    1. Applies average pooling with kernel size 2x2 and stride 2.
    2. Performs singular value decomposition (SVD) on the result.
    3. Returns the first singular value of the SVD.
    """
    pooled = F.avg_pool2d(input_tensor, kernel_size=2, stride=2)
    u, s, v = torch.linalg.svd(pooled)
    return s[0]

function_signature = {
    "name": "my_complex_function",
    "inputs": [
        ((4, 4, 4, 4), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
