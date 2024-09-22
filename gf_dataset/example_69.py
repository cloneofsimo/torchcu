
import torch
import torch.nn.functional as F

def torch_laplace_filter(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies a Laplacian filter to an input tensor using a 3x3 kernel.
    """
    kernel = torch.tensor([[0.0, -1.0, 0.0],
                           [-1.0, 4.0, -1.0],
                           [0.0, -1.0, 0.0]], dtype=torch.float32)
    return F.conv2d(input_tensor, kernel.unsqueeze(0).unsqueeze(0), padding=1)

function_signature = {
    "name": "torch_laplace_filter",
    "inputs": [
        ((1, 1, 4, 4), torch.float32),
    ],
    "outputs": [
        ((1, 1, 4, 4), torch.float32),
    ]
}
