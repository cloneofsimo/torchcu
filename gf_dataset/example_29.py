
import torch
import torch.nn.functional as F

def torch_distance_transform_qr_backward(input_tensor: torch.Tensor, target_tensor: torch.Tensor, p: int = 2) -> torch.Tensor:
    """
    Calculates the distance transform between input and target tensors, applies QR decomposition,
    and performs backward pass through the QR decomposition.
    """
    # Calculate distance transform
    distance = F.pdist(input_tensor, p=p)
    
    # Apply QR decomposition
    q, r = torch.linalg.qr(distance)

    # Backward pass (calculate gradient with respect to input)
    grad_input = torch.matmul(q, torch.matmul(r, target_tensor))
    
    return grad_input

function_signature = {
    "name": "torch_distance_transform_qr_backward",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
