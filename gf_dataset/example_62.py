
import torch
import torch.nn.functional as F

def torch_3d_hamming_maxpool(input_tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Performs 3D max pooling and then calculates pairwise Hamming distance between pooled features.
    """
    pooled = F.max_pool3d(input_tensor, kernel_size=kernel_size)
    distances = torch.cdist(pooled.flatten(start_dim=1), pooled.flatten(start_dim=1), p=1)  # Hamming distance (p=1)
    return distances

function_signature = {
    "name": "torch_3d_hamming_maxpool",
    "inputs": [
        ((10, 64, 16, 16, 16), torch.float32),
        (3, )  # kernel_size
    ],
    "outputs": [
        ((10, 10), torch.float32),
    ]
}
