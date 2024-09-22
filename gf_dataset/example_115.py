
import torch
import torch.nn.functional as F

def laplace_filter_hessian(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies a Laplacian filter to the input tensor and then computes the Hessian matrix.
    """
    # Apply Laplacian filter
    laplace_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
    laplace_output = F.conv2d(input_tensor.unsqueeze(1), laplace_kernel.unsqueeze(0).unsqueeze(0), padding=1)

    # Compute Hessian matrix using finite differences
    h_x = torch.zeros_like(laplace_output)
    h_y = torch.zeros_like(laplace_output)
    h_x[:, :, 1:, :] = laplace_output[:, :, 1:, :] - laplace_output[:, :, :-1, :]
    h_y[:, :, :, 1:] = laplace_output[:, :, :, 1:] - laplace_output[:, :, :, :-1]
    hessian = torch.stack([h_x, h_y], dim=1)

    return hessian

function_signature = {
    "name": "laplace_filter_hessian",
    "inputs": [
        ((1, 1, 10, 10), torch.float32),
    ],
    "outputs": [
        ((1, 2, 10, 10), torch.float32),
    ]
}
