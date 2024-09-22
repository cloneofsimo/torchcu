
import torch

def torch_rms_energy_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Calculate root mean square energy of the input tensor, incorporating a weighting factor.
    """
    output = torch.einsum('ij,kl->ikl', input_tensor, weight)  # Transpose and multiply
    output.sqrt_()  # In-place square root
    output.square_()  # In-place square
    return output.mean(dim=(-1, -2))  # Mean over last two dimensions

function_signature = {
    "name": "torch_rms_energy_function",
    "inputs": [
        ((2, 3, 4), torch.float32),
        ((2, 3), torch.float32)
    ],
    "outputs": [
        ((2,), torch.float32)
    ]
}
