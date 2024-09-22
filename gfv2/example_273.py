
import torch
import torch.nn.functional as F

def custom_function(input_tensor: torch.Tensor, weights: torch.Tensor, padding: int, gather_index: torch.Tensor) -> torch.Tensor:
    """
    This function performs several operations on the input tensor:
    1. Reflect padding.
    2. Element-wise maximum with a weight tensor.
    3. Gather operation based on gather_index.
    4. Calculates Wasserstein distance between the gathered tensor and the original input.
    5. Returns the Wasserstein distance tensor.
    """
    # Padding
    padded_tensor = F.pad(input_tensor, (padding, padding, padding, padding), mode='reflect')
    
    # Element-wise maximum
    max_tensor = torch.maximum(padded_tensor, weights)
    
    # Gather
    gathered_tensor = torch.gather(max_tensor, 1, gather_index.unsqueeze(1))
    
    # Wasserstein distance
    wasserstein_dist = torch.cdist(gathered_tensor, input_tensor, p=1)
    
    return wasserstein_dist

function_signature = {
    "name": "custom_function",
    "inputs": [
        ((10, 10), torch.float32),
        ((10, 10), torch.float32),
        (1, torch.int32),
        ((10,), torch.int64)
    ],
    "outputs": [
        ((10, 10), torch.float32)
    ]
}
