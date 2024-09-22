
import torch

def frobenius_norm_with_meshgrid(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Frobenius norm of a tensor using meshgrid and bfloat16 computation.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    grid_x, grid_y = torch.meshgrid(torch.arange(input_tensor.shape[0]), torch.arange(input_tensor.shape[1]))
    indices = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    input_bf16_flat = input_bf16.flatten()
    selected_values = torch.gather(input_bf16_flat, 0, indices.flatten())
    squared_sum = torch.sum(selected_values * selected_values)
    return torch.sqrt(squared_sum).to(torch.float32)

function_signature = {
    "name": "frobenius_norm_with_meshgrid",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}
