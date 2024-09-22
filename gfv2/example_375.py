
import torch

def my_function(input_tensor: torch.Tensor, gt: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes a score based on pairwise distances between input and target tensors, compared to a ground truth.
    """
    # Ensure all tensors are in fp16 for efficiency
    input_tensor = input_tensor.to(torch.float16)
    gt = gt.to(torch.float16)
    target_tensor = target_tensor.to(torch.float16)

    # Calculate pairwise Euclidean distances between input and target tensors
    distances = torch.cdist(input_tensor, target_tensor, p=2)

    # Compute element-wise difference between distances and ground truth
    diff = torch.abs(distances - gt.unsqueeze(0))

    # Sum across the second dimension (distance pairs) and apply sigmoid
    score = torch.sigmoid(torch.sum(diff, dim=1))

    return score

function_signature = {
    "name": "my_function",
    "inputs": [
        ((1, 4), torch.float32),
        ((1, 4), torch.float32),
        ((1, 4), torch.float32),
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
