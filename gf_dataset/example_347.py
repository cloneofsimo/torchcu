
import torch
import torch.nn.functional as F

def torch_eye_backward(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the gradient of the identity matrix with respect to the input tensor.
    """
    eye_tensor = torch.eye(input_tensor.size(0), dtype=torch.float32, device=input_tensor.device)
    return torch.autograd.grad(eye_tensor.sum(), input_tensor, create_graph=True)[0]

function_signature = {
    "name": "torch_eye_backward",
    "inputs": [
        ((4, 4), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
