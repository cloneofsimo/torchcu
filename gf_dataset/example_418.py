
import torch
import torch.nn.functional as F

def torch_scatter_avgpool_function(input_tensor: torch.Tensor, indices: torch.Tensor, dim: int, kernel_size: int) -> torch.Tensor:
    """
    Performs scatter operation followed by average pooling.
    """
    scattered_tensor = torch.scatter_add(torch.zeros_like(input_tensor), dim, indices, input_tensor)
    pooled_tensor = F.avg_pool1d(scattered_tensor.unsqueeze(1), kernel_size=kernel_size, stride=kernel_size).squeeze(1)
    return pooled_tensor.to(torch.float32)  # Return in FP32

function_signature = {
    "name": "torch_scatter_avgpool_function",
    "inputs": [
        ((10, 5), torch.float32),
        ((10,), torch.int64),
        (0, None),
        (3, None),
    ],
    "outputs": [
        ((10, 5), torch.float32),
    ]
}
