
import torch

def roberts_mish_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies Roberts cross-gradient operator, Mish activation, and sums along all dimensions.
    """
    # Roberts cross-gradient
    dy = input_tensor[:, 1:, :] - input_tensor[:, :-1, :]
    dx = input_tensor[:, :, 1:] - input_tensor[:, :, :-1]
    grad_x = dx[:, :-1, :-1]
    grad_y = dy[:, :-1, :-1]

    # Mish activation
    mish = grad_x * torch.tanh(torch.nn.functional.softplus(grad_x)) + grad_y * torch.tanh(torch.nn.functional.softplus(grad_y))

    # Sum along all dimensions
    output = torch.sum(mish)
    return output

function_signature = {
    "name": "roberts_mish_sum",
    "inputs": [
        ((10, 10, 10), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
