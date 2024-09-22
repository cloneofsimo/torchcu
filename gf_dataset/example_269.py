
import torch

def torch_round_maxpool_topk_function(input_tensor: torch.Tensor, kernel_size: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs rounding, 3D max pooling, top-k operation, and backpropagation.
    """
    rounded_tensor = torch.round(input_tensor)
    pooled_tensor = torch.nn.functional.max_pool3d(rounded_tensor, kernel_size=kernel_size)
    values, indices = torch.topk(pooled_tensor.flatten(), k)
    
    # Backpropagation
    values.backward()
    
    return values, indices

function_signature = {
    "name": "torch_round_maxpool_topk_function",
    "inputs": [
        ((16, 3, 10, 10, 10), torch.float32),
        (3,), torch.int32,
        (10,), torch.int32
    ],
    "outputs": [
        ((10,), torch.float32),
        ((10,), torch.int64)
    ]
}
