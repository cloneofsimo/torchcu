
import torch

class SpectralRolloffModule(torch.nn.Module):
    def __init__(self, rolloff_freq: float = 0.85):
        super().__init__()
        self.rolloff_freq = rolloff_freq

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculates the spectral rolloff of the input tensor along the last dimension.
        """
        # Calculate the spectral rolloff
        spectral_rolloff = torch.mean(input_tensor, dim=-1) * self.rolloff_freq

        # Return the rolloff values
        return spectral_rolloff

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Calculates the gradients for the spectral rolloff operation.
        """
        # Since spectral rolloff is a mean operation, gradients are equally distributed
        # across all elements along the last dimension.
        grad_input = grad_output.unsqueeze(-1) / input_tensor.size(-1)

        # Return the gradients
        return grad_input

function_signature = {
    "name": "spectral_rolloff_module",
    "inputs": [
        ((10, 16), torch.float32)
    ],
    "outputs": [
        ((10,), torch.float32)
    ]
}
